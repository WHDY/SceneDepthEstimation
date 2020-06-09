from PIL import Image, ImageTk
import tkinter as tk
import tkinter.messagebox
from tkinter import filedialog
from tkinter import simpledialog
import paramiko
import numpy as np
import re

class Depth_estimation_application(tk.Frame):
    def __init__(self, master=None, myssh=None):
        super().__init__(master)
        self.master = master
        self.pack()

        self.ssh = myssh
        self.sftp=None
        self.host = None
        self.port = None
        self.user = None
        self.password = None

        self.left_image_path = None
        self.right_image_path = None
        self.depth_image_path = None
        self.image_number = 1
        self.left_image = None
        self.left_image_tk = None

        self.image = None
        self.image_tk = None
        self.image_data = None
        self.path = None

        self.detected_object = None
        self.detected_object_tk = None

        self.is_select_area = False
        self.is_first = True
        self.last_select_rect = None
        self.initial_x = 0
        self.initial_y = 0
        self.end_x = 0
        self.end_y = 0

        self.createWidget()
        self.create_server_relevant()

    def createWidget(self):

        # 画布， 并显示初始图片
        self.initial_image = Image.open(r'D:\TensorFlow_Practice\mc_cnn_fst\GUI\depth_image.png')
        self.initial_image_tk = ImageTk.PhotoImage(self.initial_image)
        self.Canvas = tk.Canvas(self.master, width=376, height=240, bg='white', bd=0)
        self.Canvas.create_image(188, 120, anchor='center', image=self.initial_image_tk)
        self.Canvas.place(x=0, y=250)
        self.Canvas.bind('<B1-Motion>', self.select_by_rect)
        self.Canvas.bind('<ButtonRelease-1>', self.select_done)

        self.original_image = Image.open(r'D:\TensorFlow_Practice\mc_cnn_fst\GUI\original_image.png')
        self.original_image_tk = ImageTk.PhotoImage(self.original_image)
        self.Canvas1 = tk.Canvas(self.master, width=376, height=240, bg='white', bd=0)
        self.Canvas1.create_image(188, 120, anchor='center', image=self.original_image_tk)
        self.Canvas1.place(x=0, y=0)

        # 选择深度图
        self.path_tk = tk.StringVar()
        self.select_depth_button = tk.Button(self.master, text='本地图片选择', width=20, height=1, command=self.select_path)
        self.select_depth_button.place(x=395, y=255)
        self.path_lable = tk.Label(self.master, text='路径', width=5, height=1)
        self.path_lable.place(x=395, y=295)
        self.path_show = tk.Entry(self.master, width=30, textvariable=self.path_tk, bd=2)
        self.path_show.place(x=435, y=295)
        self.image_show_button = tk.Button(self.master, text='确定', width=10, height=1, command=self.image_show)
        self.image_show_button.place(x=565, y=255)

        # 选择需要估计深度的物体
        self.select_estimate_area = tk.Button(self.master, text='选择需要估计深度的物体', width=20, height=1,
                                              command=self.select_estimate_area)
        self.select_estimate_area.place(x=395, y=345)
        self.estimate = tk.Button(self.master, text='确定', width=10, height=1, command=self.estimate)
        self.estimate.place(x=565, y=345)

        # 深度信息显示
        self.depth_tk = tk.StringVar()
        self.depth_label = tk.Label(self.master, text='所选物体深度为', width=15, height=1)
        self.depth_label.place(x=392, y=385)
        self.depth_show = tk.Entry(self.master, textvariable=self.depth_tk, bd=2)
        self.depth_show.place(x=500, y=385)

        # 退出
        self.btnQuit = tk.Button(self.master, text='退出', width=20, height=1, command=root.destroy)
        self.btnQuit.place(x=500, y=455)

    def select_path(self):
        self.path = filedialog.askopenfilename()
        self.path_tk.set(self.path)

    def image_show(self):
        self.image = Image.open(self.path).convert('L')
        width, height = self.image.size
        self.image = self.image.resize((width//2, height//2))
        self.image_data = self.image.load()
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.Canvas.create_image(188, 120, anchor='center', image=self.image_tk)

    def select_estimate_area(self):
        self.is_select_area = True

    def estimate(self):
        # self.is_select_area = False
        sum = 0
        number = 0
        probability = np.zeros(shape=[255], dtype=np.float32)

        width = self.end_x - self.initial_x + 1
        height = self.end_y - self.initial_y + 1

        for j in range(self.initial_y, self.end_y + 1):
            for i in range(self.initial_x, self.end_x + 1):
                probability[self.image_data[i, j]] += 1

        target = np.argmax(probability)

        self.detected_object = Image.new('RGB', (width, height))
        self.detected_object_data = self.detected_object.load()
        for j in range(self.initial_y, self.end_y + 1):
            for i in range(self.initial_x, self.end_x + 1):
                intensity = self.image_data[i, j]
                if abs(target - intensity) < 6:
                    self.detected_object_data[i - self.initial_x, j - self.initial_y] = (0, 255, 0)
                    sum += self.image_data[i, j] / 2
                    number += 1
                else:
                    self.detected_object_data[i - self.initial_x, j - self.initial_y] = (intensity, intensity, intensity)
        self.detected_object_tk = ImageTk.PhotoImage(self.detected_object)
        self.Canvas.create_image(self.initial_x+width//2, self.initial_y+height//2, image=self.detected_object_tk)
        self.depth_tk.set('{}m'.format(round(45.569/(sum/number), 2)))

    def select_by_rect(self, event):
        if self.is_select_area is True:
            self.Canvas.delete(self.last_select_rect)
            if self.is_first is True:
                self.initial_x = event.x
                self.initial_y = event.y
                self.is_first = False
            self.last_select_rect = self.Canvas.create_rectangle(self.initial_x, self.initial_y, event.x, event.y,
                                                                 outline='#00ff00')

    def select_done(self, event):
        self.is_first = True
        self.end_x = event.x
        self.end_y = event.y
        if self.initial_x > self.end_x:
            tmp = self.initial_x
            self.initial_x = self.end_x
            self.end_x = tmp

        if self.initial_y > self.end_y:
            tmp = self.initial_y
            self.initial_y = self.end_y
            self.end_y = tmp

    def create_server_relevant(self):
        self.connect_server_button = tk.Button(self.master, text='连接服务器', width=20, height=1, command=self.require_info)
        self.connect_server_button.place(x=395, y=20)

        self.upload_button = tk.Button(self.master, text='上传左图', width=15, height=1, command=self.uploadL)
        self.upload_button.place(x=395, y=60)

        self.download_button = tk.Button(self.master, text='上传右图', width=15, height=1, command=self.uploadR)
        self.download_button.place(x=530, y=60)

        self.calculate_button = tk.Button(self.master, text='计算深度图', width=30, height=1, command=self.calculate)
        self.calculate_button.place(x=395, y=100)

        self.receive_button = tk.Button(self.master, text='获取深度图', width=30, height=1, command=self.receive)
        self.receive_button.place(x=395, y=140)

    def require_info(self):
        self.require_info_dialog_box = tk.Toplevel()
        self.require_info_dialog_box.title('连接服务器')
        self.require_info_dialog_box.geometry('310x150+450+250')

        host_id_label = tk.Label(self.require_info_dialog_box, text='服务器ID:')
        host_id_label.place(x=20, y=20)
        self.host_id_tk = tk.StringVar()
        host_id_entry = tk.Entry(self.require_info_dialog_box, textvariable=self.host_id_tk, width=15, bd=2)
        host_id_entry.place(x=80, y=20)

        port_label = tk.Label(self.require_info_dialog_box, text='端口号:')
        port_label.place(x=200, y=20)
        self.port_tk = tk.StringVar()
        port_entry = tk.Entry(self.require_info_dialog_box, textvariable=self.port_tk, width=3, bd=2)
        port_entry.place(x=250, y=20)

        user_name_label = tk.Label(self.require_info_dialog_box, text='用户名:')
        user_name_label.place(x=20, y=60)
        self.user_name_tk = tk.StringVar()
        user_name_entry = tk.Entry(self.require_info_dialog_box, textvariable=self.user_name_tk, width=8, bd=2)
        user_name_entry.place(x=80, y=60)

        password_label = tk.Label(self.require_info_dialog_box, text='密码:')
        password_label.place(x=160, y=60)
        self.password_tk = tk.StringVar()
        password_entry = tk.Entry(self.require_info_dialog_box, textvariable=self.password_tk, show='*', width=10, bd=2)
        password_entry.place(x=200, y=60)

        connect_button = tk.Button(self.require_info_dialog_box, text='连接', width=15, height=1, command=self.connect)
        connect_button = connect_button.place(x=20, y=100)
        cancel_button = tk.Button(self.require_info_dialog_box, text='取消', width=15, height=1, command=self.cancel)
        cancel_button = cancel_button.place(x=160, y=100)

    def connect(self):
        self.host = self.host_id_tk.get()
        self.port = self.port_tk.get()
        self.user = self.user_name_tk.get()
        self.password = self.password_tk.get()
        ssh.connect(hostname=self.host, port=self.port, username=self.user, password=self.password)

        trans = paramiko.Transport(sock=(self.host))
        trans.connect(username=self.user, password=self.password)
        self.sftp = paramiko.SFTPClient.from_transport(trans)

        stdin, stdout, stderr = ssh.exec_command('rm -r ./mc_cnn_fst/UI_use')
        stdin, stdout, stderr = ssh.exec_command('rm -r ./mc_cnn_fst/UI_disparity')

        stdin, stdout, stderr = ssh.exec_command('mkdir ./mc_cnn_fst/UI_use')
        stdin, stdout, stderr = ssh.exec_command('mkdir ./mc_cnn_fst/UI_disparity')

        self.require_info_dialog_box.destroy()
        tkinter.messagebox.showinfo("连接信息", "连接成功！")

    def cancel(self):
        self.require_info_dialog_box.destroy()

    def uploadL(self):
        self.left_image_path = filedialog.askopenfilename()
        self.image_number = re.sub("\D", "", self.left_image_path)
        # print(self.image_number)
        self.sftp.put(self.left_image_path, '/home/rjt1/mc_cnn_fst/UI_use/left_{}.jpg'.format(self.image_number))

        self.left_image = Image.open(self.left_image_path).convert('L')
        # print(self.left_image_path)
        width, height = self.left_image.size
        self.left_image = self.left_image.resize((width // 2, height // 2))
        self.left_image_tk = ImageTk.PhotoImage(self.left_image)
        self.Canvas1.create_image(188, 120, anchor='center', image=self.left_image_tk)

    def uploadR(self):
        self.right_image_path = filedialog.askopenfilename()
        self.sftp.put(self.right_image_path, '/home/rjt1/mc_cnn_fst/UI_use/right_{}.jpg'.format(self.image_number))

    def calculate(self):
        stdin, stdout, stderr = ssh.exec_command('conda activate condah')
        stdin, stdout, stderr = ssh.exec_command('python ./mc_cnn_fst/match_single_ui.py -i {}'.format(self.image_number))
        '''
        stdin, stdout, stderr = ssh.exec_command(
            'cp ./mc_cnn_fst/disparity/ld{}.png ./mc_cnn_fst/UI_disparity/ld{}.png'.format(self.image_number,
                                                                                           self.image_number))
        '''

    def receive(self):
        self.depth_image_path =  'D:\TensorFlow_Practice\mc_cnn_fst\GUI\ld{}.png'.format(self.image_number)
        self.sftp.get('/home/rjt1/mc_cnn_fst/UI_disparity/ld{}.png'.format(self.image_number), self.depth_image_path)
        # self.image_number += 1

        self.image = Image.open(self.depth_image_path)
        width, height = self.image.size
        self.image = self.image.resize((width//2, height//2))
        self.image_data = self.image.load()
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.Canvas.create_image(188, 120, anchor='center', image=self.image_tk)


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("660x500+400+200")
    root.title('Depth Estimation Application')
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    app = Depth_estimation_application(master=root, myssh=ssh)

    root.mainloop()
