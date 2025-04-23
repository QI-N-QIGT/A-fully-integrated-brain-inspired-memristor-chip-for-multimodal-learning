#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <signal.h> // For handling Ctrl+C

class LidarDataSaver
{
public:
    LidarDataSaver()
    {
        // 订阅雷达数据的话题
        sub = nh.subscribe("/scan", 1000, &LidarDataSaver::scanCallback, this);
        // 创建rosbag文件
        bag_name = "lidar_data.bag";
        bag.open(bag_name, rosbag::bagmode::Write);

        // 设置Ctrl+C信号处理函数
        signal(SIGINT, &LidarDataSaver::handleSigInt);//也可以不加，ros系统自己有该信号的处理
    }

    ~LidarDataSaver()
    {
        // 关闭rosbag文件
        bag.close();
    }

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
    {
        // 将雷达数据保存到rosbag文件
        bag.write("/scan", msg->header.stamp, *msg);
    }

    // 处理Ctrl+C信号
    static void handleSigInt(int sig)
    {
        ROS_INFO("Received Ctrl+C. Exiting gracefully...");
        ros::shutdown();
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber sub;
    rosbag::Bag bag;
    std::string bag_name;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar_data_saver");
    LidarDataSaver lidar_data_saver;
    ros::spin(); // 等待回调函数
    return 0;
}

#if 0
class LidarDataSaver
{
    public:
        LidarDataSaver()
        {
            // 订阅雷达数据的话题
            sub = nh.subscribe("/scan", 1000, &LidarDataSaver::scanCallback, this);
            // 创建rosbag文件
            bag_name = "lidar_data.bag";
            bag.open(bag_name, rosbag::bagmode::Write);
        }
        ~LidarDataSaver()
        {
            // 关闭rosbag文件
            bag.close();
        }
        void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
        {
            // 将雷达数据保存到rosbag文件
            bag.write("/scan", ros::Time::now(), *msg);
        }
    private:
        ros::NodeHandle nh;
        ros::Subscriber sub;
        rosbag::Bag bag;
        std::string bag_name;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar_data_saver");
    LidarDataSaver lidar_data_saver;
    ros::Rate rate(10.0); // 10Hz
    while (ros::ok())
    {
        rate.sleep();
        ros::spinOnce();
    }
    return 0;
}
#endif
