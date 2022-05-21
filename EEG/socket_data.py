# coding=utf-8
import socket
import time
import qi
import time
import random
import almath
from colorama import Fore, init
init(autoreset=True)
# pepper部分：服务端，等待eeg部分发送识别结果


class speak:
    def __init__(self):
        qiApp = qi.Application()
        qiApp.start()
        # self.MI=mi
        self.get_service(qiApp)
        self.set_parameter()
        self.disable_auto_mode()
        # self.speak()
        # self.__del__()
        # qiApp.stop()

    def get_service(self, qiApp):
        # 声明需要的服务
        self.TextToSpeech = qiApp.session.service("ALTextToSpeech")
        self.AutonomousLife = qiApp.session.service("ALAutonomousLife")
        self.RobotPosture = qiApp.session.service("ALRobotPosture")
        self.AnimatedSpeech = qiApp.session.service("ALAnimatedSpeech")
        self.Motion = qiApp.session.service("ALMotion")

    def set_parameter(self):
        # 将机器人的语音设置为英文，设置机器人语音的速度
        self.TextToSpeech.setParameter("speed", 85.0)
        self.TextToSpeech.setLanguage("English")
        # 设置机器人说话的语言和语速,设置机器人头部固定不随着移动变化
        self.Motion.setStiffnesses("Head", 1.0)
        # self.TextToSpeech.setLanguage("English")

    def disable_auto_mode(self):
        # 取消机器人的自主模式，让机器人不会随着人转头
        print(Fore.GREEN + u"[I]: 取消自主模式中……")
        if self.AutonomousLife.getState() != "disabled":
            self.AutonomousLife.setState("disabled")
        # 取消了自主模式，机器人会低头，通过站立初始化让机器人抬起头
        self.RobotPosture.goToPosture("StandInit", 0.5)

    def standinit(self):
        # 取消了自主模式，机器人会低头，通过站立初始化让机器人抬起头
        self.RobotPosture.goToPosture("StandInit", 0.5)

    def speak(self, MI):
        self.Motion.setAngles("HeadYaw", 0.0 * almath.TO_RAD, 0.1)
        # self.RobotPosture.goToPosture("StandInit", 0.5)
        # self.Motion.setAngles("LShoulderPitch", 0.0 * almath.TO_RAD, 0.2)
        # self.Motion.setAngles("LShoulderRoll", 0.0 * almath.TO_RAD, 0.2)
        print(Fore.GREEN + u"[I]: 开始说话部分")
        # 说话功能的主体
        if MI == '0':
            print('rest rest')
            self.TextToSpeech.setParameter("speed", 85.0)
            self.TextToSpeech.setLanguage("Chinese")
            self.TextToSpeech.say("休息")
            # time.sleep(1)
            self.Motion.moveInit()
            self.Motion.setAngles("HeadYaw", -30.0*almath.TO_RAD, 0.1)
            # self.AnimatedSpeech.say("the emotion is neutral")
            # time.sleep(1)
        elif MI == '1':
            print('move move')
            # text=['想象左手运动', '你好呀', 'left', 'move']
            text=['想象左手运动', '你好呀']
            # n = random.randint(0, 3)
            n = random.randint(0, 1)
            self.TextToSpeech.setParameter("speed", 85.0)
            self.TextToSpeech.setLanguage("Chinese")
            # self.TextToSpeech.say("left")
            self.TextToSpeech.say(text[n])
            # time.sleep(1)
            if n==1:
                self.Motion.moveInit()
                # self.Motion.setAngles("HeadYaw", 30.0*almath.TO_RAD, 0.1)
                self.Motion.setAngles("LShoulderRoll", 20.0 * almath.TO_RAD, 0.4)
                self.Motion.setAngles("LShoulderPitch", 0.0 * almath.TO_RAD, 0.4)
                self.Motion.setAngles("LElbowRoll", -60.0 * almath.TO_RAD, 0.4)
                self.Motion.setAngles("LWristYaw", 70.0 * almath.TO_RAD, 0.4)

                # self.AnimatedSpeech.say("you are sad")
                time.sleep(1)

            elif n==0:
                self.Motion.moveInit()
                # self.Motion.setAngles("HeadYaw", 30.0*almath.TO_RAD, 0.1)
                self.Motion.setAngles("LShoulderPitch", 60.0 * almath.TO_RAD, 0.25)
                self.Motion.setAngles("LShoulderRoll", 45.0 * almath.TO_RAD, 0.25)
                # self.AnimatedSpeech.say("you are sad")
                time.sleep(0.5)

            self.RobotPosture.goToPosture("StandInit", 0.5)

    def __del__(self):
        print(Fore.GREEN + u"[I]: 程序运行结束")
        # qiApp.stop()


if __name__ == "__main__":
    #IPV4,TCP协议
    sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    #绑定ip和端口，bind接受的是一个元组
    sock.bind(('192.168.43.138',5050))
    #设置监听，其值阻塞队列长度，一共可以有5+1个客户端和服务器连接
    sock.listen(5)
    # qiApp = qi.Application()
    pepper_MI=speak()

    while True:
        # 将发送数据转化为String
        # s=str(a)
        # 等待客户请求
        connection, address = sock.accept()

        # 打印客户端地址
        print("client ip is:", address)
        # 接收数据,并存入buf
        buf = connection.recv(40960)
        # print(buf.decode('utf-8'))
        # print(type(buf.decode('utf-8')))
        result = buf.decode('utf-8')
        print(result)

        pepper_MI.speak(result)  # 注意看看buf要做什么修改
        # pepper_MI.standinit()

        # 发送数据
        # connection.send(bytes(s, encoding="utf-8"))
        # 关闭连接
        connection.close()
        time.sleep(1)
    # 关闭服务器
    sock.close()
    pepper_MI.standinit()

    # qiApp.stop()
    # speak_intance = speak(buf) # 注意看看buf要做什么修改
