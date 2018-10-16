import glob, os
import tkinter as tk
import numpy as np
from tkinter.filedialog import askdirectory
from PIL import ImageTk as itk
from PIL import Image

#Testing out git reverting and versions#

#*****MAIN CLASS*****#
class main_application(tk.Frame):

    def __init__(self, master=None, **kwargs):
        global pics
        global pics_ind
       
        #MASTER FRAME THINGS HERE
        tk.Frame.__init__(self, master)            
        self.master = master
       
        self.master.title("FRB viewer")
        self.grid(row=0,column=0,rowspan = 15, columnspan = 15)
        self.master.configure(background = 'black')
        self.master.grid_rowconfigure(0, weight = 1)
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
       
       
        #IMAGE
        self.pngimage = Image.open(pics[pics_ind])
        width, height = self.pngimage.size
        self.pngimage = self.pngimage.resize((width,height), Image.ANTIALIAS)
        self.img = itk.PhotoImage(self.pngimage)
        self.canvas = tk.Canvas(self,width = width, height = height)
        self.img1 = self.canvas.create_image(0,0,image = self.img, anchor = "nw")
        self.canvas.grid(row = 0, column = 0, columnspan = 10, rowspan = 10)
       
       
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
      
   
    #Method to change the img on the canvas
    def changeimg(self): 
        self.pngimage = Image.open(pics[pics_ind])
        self.img = itk.PhotoImage(self.pngimage)
        self.canvas.itemconfig(self.img1, image = self.img)
       
    #Changing image with left or right key
    def key(self, event):
        global pics_ind
        global pics
      
        if event.keysym == 'Right':
            if (pics_ind >= (len(pics) - 1)):
                print("Reached end of files.")
            else:
                pics_ind += 1
                self.changeimg()
        elif event.keysym == 'Left':
            if (pics_ind <= 0):
                print("At the first file already.")
            else:
                pics_ind -= 1
                self.changeimg()
       
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
            print(pics[pics_ind].split("\\")[9])
            os.rename(pics[pics_ind],folder + "\\frb\\" + pics[pics_ind].split("\\")[9])
            result_file.write(folder + "\\frb\\" + pics[pics_ind].split("\\")[9] + "\n")
        else:
            print(pics[pics_ind].split("\\")[9])
            os.rename(pics[pics_ind],folder + "\\no_frb\\" + pics[pics_ind].split("\\")[9])
            result_file.write(folder + "\\no_frb\\" + pics[pics_ind].split("\\")[9] + "\n")
  
        #Moves to next img
        if ((pics_ind >= (len(pics) - 1)) and (len(pics) != 1)):
            if (len(pics) != 0):
                pics_ind -= 1 
                self.changeimg()
                del pics[pics_ind + 1]
            print("Reached end of files.")
        elif (len(pics) == 1):
            self.canvas.delete("all")
        else:
            pics_ind += 1
            self.changeimg()
            del pics[pics_ind - 1]
            pics_ind -= 1 
 
def main():
    global pics_ind
    global pics
    global result_file
    global app
    global idir_path
    global odir_path
    
    idir_path = os.getcwd() + "\\idir"
    odir_path = os.getcwd() + "\\odir"

    completeName = os.path.join(odir_path,"results.txt")
    result_file=open(completeName,"a")
   
    pics = []       #List of filenames to be viewed in program
    pics_ind = 0    #Index for the img currently being viewed
   
    #Adds names of files ending with .gif in 'pics' list
    for file in glob.glob(idir_path + "/" + "*.png"):
        pics.append(file)
   
    root = tk.Tk()
    #root.geometry("500x500")
    app = main_application(root)
    root.bind('<Key>', lambda event: app.key(event))
   
    root.mainloop()
   
    result_file.close()
   
if __name__ == '__main__':
    main()