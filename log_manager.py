from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import QTime, QDateTime
from PyQt5.QtGui import QTextCursor
import os

class log_manager():
    def __init__(self):
        if not os.path.exists('_log'):
            os.mkdir('_log')
        # self.__date_and_time = QDateTime.currentDateTime().toString("yyMMdd-hh-mm-ss")
        self.__date = QDateTime.currentDateTime().toString("yyyy-MM-dd")
        self.__log_file_name = '_log/' + self.__date + '.html'

        if not os.path.exists('_log/log_styles.css'):
            f = open('_log/log_styles.css', 'w')
            f.write('*{margin:0px}')
            f.write("p{font-family: 'Spoqa Han Sans', snan-serif;}")
            f.write('body{background-color:#292D3E}')
            f.close()

        if not os.path.isfile(self.__log_file_name):
            f = open(self.__log_file_name, 'a')
            f.write('<link rel="stylesheet" href="./log_styles.css"><link>')
            f.write('<link rel="stylesheet" href="https://spoqa.github.io/spoqa-han-sans/css/SpoqaHanSans-kr.css" type="text/css">')
            f.close()

    def print_log(self, log, edit, color='white', time=True, hide=False):
        self.__log_write_html(log, edit, color, time, hide)
        

    def print_blank(self):
        self.__write_log_file('<br />')

    def __log_write_html(self, log, edit, color, time, hide):
        log_for_write = self.__log_with_time(log) if time == True else log

        style_font = '<font color={0}>{1}</font>'.format(color, log_for_write)
        self.__write_log_file('<p>{}</p>'.format(style_font))

        if not hide:
            edit.append(style_font)
            edit.moveCursor(QTextCursor.End)


    def __log_with_time(self, log):
        time = QTime.currentTime().toString('hh:mm:ss:zzz')
        final_log = '['+time+'] ' + log
        return final_log

    def __write_log_file(self, log):
        f = open(self.__log_file_name, 'a')
        f.write(log)
        f.close()