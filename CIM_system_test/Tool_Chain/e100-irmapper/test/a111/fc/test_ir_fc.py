from unittest import TestCase, main
from e100_irmapper.device.a111 import A111NpuDevice, A111TileDevice
from e100_irtool import make_device


class TestA111Device(TestCase):

    def test_a111_dev(self):
        dev = make_device('a111-npu')
        self.assertTrue(isinstance(dev, A111NpuDevice))
        self.assertTrue(isinstance(dev.devices['tile'], A111TileDevice))


if __name__ == '__main__':
    main()