import multiprocessing

class MultiProcessRuntime:

    def __init__(self, runtime, config, process_num=4, simulation=True):
        self.runtime = runtime
        self.config = config
        self.process_num = process_num
        self.simulation = simulation

    def calculate(self, ir, inputs, kwargs):
        rt = self.runtime(**self.config)
        rt.init_rpc(ir, simulation=self.simulation)
        result = getattr(rt, 'run_ir')(ir, inputs, **kwargs)
        return result

    def run_data_parallel(self, mapped_ir, in_value, *, weights=None, outputs=None, callback=None, logfile=None):
        assert len(mapped_ir) == len(in_value)
        PROCESSES = self.process_num
        print()
        kwargs = {}
        if weights != None:
            kwargs.update({'weights': weights})
        if outputs != None:
            kwargs.update({'outputs': outputs})
        if callback != None:
            kwargs.update({'callback': callback})
        if logfile != None:
            kwargs.update({'logfile': logfile})
        re_value = []
        with multiprocessing.Pool(PROCESSES) as pool:
            TASKS = []
            index = 0
            for i in in_value:
                TASKS.append((mapped_ir[index], i, kwargs))
                index += 1
            assert TASKS != []
            results = [pool.apply_async(self.calculate, t) for t in TASKS]
            for r in results:
                re_value.append(r.get())
        return re_value

    def pipe_calculate(self, ir, input_q, output_q, kwargs, final):
        rt = self.runtime(**self.config)
        rt.init_rpc(ir, simulation=self.simulation)
        while True:
            inputs = input_q.get()
            if inputs is None:
                if not final:
                    output_q.put(inputs)
                break
            result = getattr(rt, 'run_ir')(ir, inputs, **kwargs)
            if final:
                output_q.append(result)
            else:
                output_q.put(result)

    def run_model_parallel(self, mapped_ir, in_value, *, weights=None, outputs=None, callback=None, logfile=None):
        assert len(outputs) == self.process_num
        kwargs_ = []
        assert len(weights) == len(outputs)
        assert len(outputs) == len(callback)
        assert len(callback) == len(logfile)
        for i in range(len(weights)):
            kwargs = {}
            if weights[i] != None:
                kwargs.update({'weights': weights[i]})
            if outputs[i] != None:
                kwargs.update({'outputs': outputs[i]})
            if callback[i] != None:
                kwargs.update({'callback': callback[i]})
            if logfile[i] != None:
                kwargs.update({'logfile': logfile[i]})
            kwargs_.append(kwargs)
        results = multiprocessing.Manager().list()
        PROCESSES = self.process_num
        print()
        queue_list = []
        for i in range(self.process_num):
            queue_list.append(multiprocessing.Queue())
        process_list = []
        for j in range(self.process_num):
            if j == self.process_num - 1:
                process_list.append(multiprocessing.Process(target=self.pipe_calculate, args=(mapped_ir, queue_list[j], results, kwargs_[j], True)))
            else:
                process_list.append(multiprocessing.Process(target=self.pipe_calculate, args=(mapped_ir, queue_list[j], queue_list[j + 1], kwargs_[j], False)))
        for i in range(self.process_num):
            process_list[i].start()
        for i in in_value:
            queue_list[0].put(i)
        queue_list[0].put(None)
        for i in range(self.process_num):
            process_list[i].join()
        for i in range(self.process_num):
            queue_list[i].close()
        return list(results)