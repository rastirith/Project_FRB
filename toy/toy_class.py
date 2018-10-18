import glob, os
import tkinter as tk
import numpy as np
#from tkinter.filedialog import askdirectory
#from PIL import ImageTk as itk
#from PIL import Image
   
#import matplotlib
#import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure











#*****MAIN CLASS*****#
class main_application(tk.Frame):

    def __init__(self, master):
        global pics
        global pics_ind
       
        #MASTER FRAME THINGS HERE
        tk.Frame.__init__(self, master)            
        self.master = master
       
        self.master.title("FRB viewer")
        self.grid(row=0,column=0)#,rowspan = 15, columnspan = 15)
        self.master.configure(background = 'black')
        self.master.grid_rowconfigure(0)
        self.master.grid_columnconfigure(0, weight = 1)
      
      
        #MENU OBJECTS HERE
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        filemenu = tk.Menu(menu)
        menu.add_cascade(label = "File", menu = filemenu)
        filemenu.add_command(label = "Choose input directory...",command = self.client_exit)
        filemenu.add_command(label = "Choose output directory...",command = self.client_exit)
        filemenu.add_separator()
        filemenu.add_command(label = "Exit", command = self.client_exit)
       
       
        #PLOT
        self.f = plotimg(pics[pics_ind])
        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 10, rowspan = 10)


        #BUTTONS
        self.frb_button = tk.Button(self,text = "FRB", command = lambda: self.frbchoice(odir_path,1))
        self.nfrb_button = tk.Button(self,text = "NO FRB", command = lambda: self.frbchoice(odir_path,0))
        self.frb_button.grid(row = 13, column = 3, pady = "20", padx = "20")
        self.nfrb_button.grid(row = 13, column = 6, pady = "20", padx = "20")
       
       
        #TEXT WINDOW
        self.T = tk.Text(self,height = 6,width = 35)
        self.S = tk.Scrollbar(self)
        self.S.config(command = self.T.yview)
        self.T.config(yscrollcommand = self.S.set)
        self.T.grid(row = 13,column = 9)
        self.S.grid(row = 13,column = 9,sticky = "E")
       
    #Changing image with left or right key
    def key(self, event):
        global pics_ind
        global pics
      
        if event.keysym == 'Right':
            if (pics_ind >= (len(pics) - 1)):
                print("Reached end of files.")
            else:
                pics_ind += 1
                self.f = plotimg(pics[pics_ind])
                self.canvas = FigureCanvasTkAgg(self.f, self)
                self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 10, rowspan = 10)
        elif event.keysym == 'Left':
            if (pics_ind <= 0):
                print("At the first file already.")
            else:
                pics_ind -= 1
                self.f = plotimg(pics[pics_ind])
                self.canvas = FigureCanvasTkAgg(self.f, self)
                self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 10, rowspan = 10)
       
    #Exits client
    def client_exit(self):
        exit()   
      
    #Method to put image in appropriate folder, make a note where that is, and change the image on display
    def frbchoice(self, folder, choice):
        global result_file
        global pics_ind
        global pics
       
        try:    #Only creates folders if they don't already exist
            os.mkdir(folder + "//frb")
            os.mkdir(folder + "//no_frb")
        except:
            pass
       
        #Adds path name and moves img to appropriate folder
        if (choice == 1):
            os.rename(pics[pics_ind],folder + "\\frb\\" + pics[pics_ind].split("\\")[-1])
            result_file.write(folder + "\\frb\\" + pics[pics_ind].split("\\")[-1] + "\n")
        else:
            os.rename(pics[pics_ind],folder + "\\no_frb\\" + pics[pics_ind].split("\\")[-1])
            result_file.write(folder + "\\no_frb\\" + pics[pics_ind].split("\\")[-1] + "\n")
  
        #Moves to next img
        if ((pics_ind >= (len(pics) - 1)) and (len(pics) != 1)):
            if (len(pics) != 0):
                pics_ind -= 1 
                self.f = plotimg(pics[pics_ind])
                self.canvas = FigureCanvasTkAgg(self.f, self)
                self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 10, rowspan = 10)
                del pics[pics_ind + 1]
            print("Reached end of files.")
        elif (len(pics) == 1):
            self.canvas.delete("all")
        else:
            pics_ind += 1
            self.f = plotimg(pics[pics_ind])
            self.canvas = FigureCanvasTkAgg(self.f, self)
            self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 10, rowspan = 10)
            del pics[pics_ind - 1]
            pics_ind -= 1 


def plotimg(path):

    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1)
    
    c = data.reshape((-1,4))
    np.savetxt('txtfilename',c)
            
    Tfile.close()
    
    columns = np.hsplit(c,4) #dm=0, time=1, ston=2, width=3

    dm = columns[0]
    time = columns[1]
    #ston = columns[2]
    width = columns[3]
    
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.scatter(time, dm, s = (width**3)/3500)
    
    return fig

 
def main():
    global pics_ind
    global pics
    global result_file
    global app
    global idir_path
    global odir_path

    idir_path = os.getcwd() + "\\idir"
    odir_path = os.getcwd() + "\\odir"
    
    try:    #Only creates folders if they don't already exist
        os.mkdir(idir_path)
        os.mkdir(odir_path)
    except:
        pass
        
    completeName = os.path.join(odir_path,"results.txt")
    result_file=open(completeName,"a")
   
    pics = []       #List of filenames to be viewed in program
    pics_ind = 0    #Index for the img currently being viewed
   
    #Adds names of files ending with .gif in 'pics' list
    for file in glob.glob(idir_path + "/" + "*.dat"):
        pics.append(file)

    root = tk.Tk()
    #plotimg(root, pics[0]).get_tk_widget().grid()
    #root.geometry("500x500")
    app = main_application(root)
    root.bind('<Key>', lambda event: app.key(event))

    root.mainloop()
   
    result_file.close()
   
if __name__ == '__main__':
    main() 