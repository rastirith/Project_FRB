import glob, os
import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import threading
#import tkFont
#from tkinter.filedialog import askdirectory
#from PIL import ImageTk as itk
#from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

fontx = ('Helvetica', 8)

#*****MAIN CLASS*****#
class start_app(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.grid()

        self.frames = {}
        for F in (main_frame, frame1):
            page_name = F.__name__
            frame = F(master=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("main_frame")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()
        frame.focus_set()
     
     
        
class main_frame(tk.Frame):
    
    def __init__(self, master, controller):
        global pics
        global pics_ind
       
        #MASTER FRAME THINGS HERE
        tk.Frame.__init__(self, master)
        self.controller = controller            
        self.controller.title("FRB viewer")
        self.grid(row=0,column=0, columnspan = 20, rowspan = 30)
        self.controller.configure(background = 'white')
        self.controller.bind('<Key>', lambda event: self.key1(event))
        
        self.menuobj()
        self.labels()
        self.textbox()
        self.buttons()
    
    # Method defining the drop down menu part of the GUI
    def menuobj(self):
        
        menu = tk.Menu(self.controller)
        self.controller.config(menu=menu)

        filemenu = tk.Menu(menu)
        menu.add_cascade(label = "File", menu = filemenu)
        filemenu.add_command(label = "Choose input directory...",command = self.client_exit)
        filemenu.add_command(label = "Choose output directory...",command = self.client_exit)
        filemenu.add_separator()
        filemenu.add_command(label = "Exit", command = self.client_exit)
        self.buttons()
        
    # Method defining the Labels (including the canvas) part of the GUI
    def labels(self):
        global xref
        global yref
        global progress
        
        self.canvasupdate(xref, yref)   #Shows the first plot of the data set as default at start-up
        tk.Label(self, text = "X:", font = "Helvetica 9 bold").grid(row = 5, column = 21)
        tk.Label(self, text = "Y:", font = "Helvetica 9 bold").grid(row = 6, column = 21)
        
        # Progress bar displaying fraction of classified files
        progress = ttk.Progressbar(self, orient="horizontal",length=100,mode='determinate')
        progress.grid(row = 21, column = 18)
        
    # Method defining the textbox part of the GUI
    def textbox(self):
        
        self.T = tk.Text(self,height = 6,width = 35)
        self.S = tk.Scrollbar(self)
        self.S.config(command = self.T.yview)
        self.T.config(yscrollcommand = self.S.set)
        self.T.grid(row = 0,column = 22, rowspan = 5, columnspan = 4, sticky = "w", padx = (0, 15))
        #self.S.grid(row = 0,column = 9, rowspan = 5, sticky = "E")
    
    # Method defining the buttons available on the GUI
    def buttons(self):
        global xref
        global yref
        
        x = tk.IntVar() # Variable value for the x-axis data set given by Radiobutton
        x.set(1)        # Sets the default plot to show time on the x-axis
        xref = x.get()
        
        y = tk.IntVar() # Variable value for the x-axis data set given by Radiobutton
        y.set(0)        # Sets the default plot to show DM on the y-axis
        yref = y.get()

        datatypes1 = [ "DM", "Time", "StoN", "Width"]
        datatypes2 = [ "DM", "Time", "StoN", "Width"]
        
        # Method called by the Radiobuttons to check and update the xref and yref values if they have changed
        def rdbchange():
            global xref
            global yref
            
            if ((y.get() != yref) or (x.get() != xref)):
                xref = x.get()
                yref = y.get()
        
        # Radiobutton allowing the user to choose what data to display on the x-axis of the plot
        for val, datatypes1 in enumerate(datatypes1):
            tk.Radiobutton(self, 
                          text=datatypes1,
                          padx = 7, 
                          variable=x, 
                          value=val,
                          command = rdbchange,
                          font = fontx).grid(row = 5, column = 22 + val)
        
        # Radiobutton allowing the user to choose what data to display on the y-axis of the plot
        for val, datatypes2 in enumerate(datatypes2):
            tk.Radiobutton(self, 
                          text=datatypes2,
                          padx = 7, 
                          variable=y, 
                          value=val,
                          command = rdbchange,
                          font = fontx).grid(row = 6, column = 22 + val)
        
        # Button to display the plot defined by user's choice of x- and y-values
        self.show_button = tk.Button(self, text = "Show", command = lambda: self.canvasupdate(xref, yref))
        self.show_button.grid(row = 7, column = 23, columnspan = 2, padx = 15)
        
        # Button to view the previously classified plots
        self.view_button = tk.Button(self, text = "View classified plots", command = lambda: self.controller.show_frame("frame1"))
        self.view_button.config(height = 1, width = 20)
        self.view_button.grid(row = 21, column = 0, padx = (10,0))
        
        # Buttons to classify whether plot shows an FRB or not.
        self.frb_button = tk.Button(self,text = "Frb", command = lambda: self.frbchoice(odir_path,1))
        self.nfrb_button = tk.Button(self,text = "No frb", command = lambda: self.frbchoice(odir_path,0))
        self.frb_button.grid(row = 1, column = 21, padx = 15)
        self.nfrb_button.grid(row = 2, column = 21, padx = 15)
        
        # Buttons to move between plots
        self.right_btn = tk.Button(self, text = "Next plot", command = lambda: self.Right())
        self.left_btn = tk.Button(self, text = "Prev. plot", command = lambda: self.Left())
        self.right_btn.grid(row = 21, column = 8, pady = 5)
        self.left_btn.grid(row = 21, column = 7, pady = 5)
        
    def Right(self):
        global pics_ind
        global pics
        global xref
        global yref
        
        if (pics_ind >= (len(pics) - 1)):  # If at the end of the array don't try to go further
            print("Reached end of files.")
        else:
            pics_ind += 1
            self.canvasupdate(xref,yref)
            
    def Left(self):
        global pics_ind
        global pics
        global xref
        global yref
        
        if (pics_ind <= 0):   # If at the beginning on the array don't try to go further back
            print("At the first file already.")
        else:
            pics_ind -= 1
            self.canvasupdate(xref,yref)
        
    #Changing image with left or right key
    def key1(self, event):
        global pics_ind
        global pics
        global xref
        global yref
        
        print("besh")
        
        if event.keysym == 'Right':
            if (pics_ind >= (len(pics) - 1)):       # If at the end of the array don't try to go further
                print("Reached end of files.")
            else:
                pics_ind += 1
                self.canvasupdate(xref,yref)
        elif event.keysym == 'Left':
            if (pics_ind <= 0):                     # If at the beginning on the array don't try to go further back
                print("At the first file already.")
            else:
                pics_ind -= 1
                self.canvasupdate(xref,yref)
     
    # Updates the progress bar    
    def bar(self):
        global length
        global progcount
        global progress
        
        progress['value']=progcount*100/length

    #Exits client
    def client_exit(self):
        exit()   
      
    #Method to put image in appropriate folder, make a note where that is, and change the image on display
    def frbchoice(self, folder, choice):
        global result_file
        global pics_ind
        global pics
        global xref
        global yref
        global progcount
        
        progcount += 1
        self.bar()
        
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
                self.canvasupdate(xref,yref)
                del pics[pics_ind + 1]
            print("Reached end of files.")
        elif (len(pics) == 1):
            self.canvas.delete("all")
        else:
            pics_ind += 1
            self.canvasupdate(xref,yref)
            del pics[pics_ind - 1]
            pics_ind -= 1 
            
    def canvasupdate(self, xref, yref):
        self.canvas = FigureCanvasTkAgg(plotimg(pics[pics_ind],xref,yref), self)
        self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 20, rowspan = 20)


class frame1(tk.Frame):
    
    def __init__(self, master, controller):
        global pics
        #global pics_ind
        global pics_ind2
       
        #MASTER FRAME THINGS HERE
        tk.Frame.__init__(self, master)
        self.controller = controller            
        self.controller.title("FRB viewer")
        self.grid(row=0,column=0, columnspan = 20, rowspan = 30)
        self.controller.configure(background = 'black')
        
        self.menuobj()
        self.labels()
        self.textbox()
        self.buttons()
        
        self.controller.bind('<Key>', lambda event: self.key(event))
    
    # Method defining the drop down menu part of the GUI
    def menuobj(self):
        
        menu = tk.Menu(self.controller)
        self.controller.config(menu=menu)

        filemenu = tk.Menu(menu)
        menu.add_cascade(label = "File", menu = filemenu)
        filemenu.add_command(label = "Choose input directory...",command = self.client_exit)
        filemenu.add_command(label = "Choose output directory...",command = self.client_exit)
        filemenu.add_separator()
        filemenu.add_command(label = "Exit", command = self.client_exit)
        self.buttons()
        
    # Method defining the Labels (including the canvas) part of the GUI
    def labels(self):
        global xref2
        global yref2
        global progress
        
        self.canvasupdate(xref2, yref2)   #Shows the first plot of the data set as default at start-up
        tk.Label(self, text = "X:", font = "Helvetica 9 bold").grid(row = 5, column = 21)
        tk.Label(self, text = "Y:", font = "Helvetica 9 bold").grid(row = 6, column = 21)
        
        # Progress bar displaying fraction of classified files
        progress = ttk.Progressbar(self, orient="horizontal",length=100,mode='determinate')
        progress.grid(row = 21, column = 18)
        
    # Method defining the textbox part of the GUI
    def textbox(self):
        
        self.T = tk.Text(self,height = 6,width = 35)
        self.S = tk.Scrollbar(self)
        self.S.config(command = self.T.yview)
        self.T.config(yscrollcommand = self.S.set)
        self.T.grid(row = 0,column = 22, rowspan = 5, columnspan = 4, sticky = "w", padx = (0, 15))
        #self.S.grid(row = 0,column = 9, rowspan = 5, sticky = "E")
    
    # Method defining the buttons available on the GUI
    def buttons(self):
        global xref2
        global yref2
        
        x = tk.IntVar() # Variable value for the x-axis data set given by Radiobutton
        x.set(1)        # Sets the default plot to show time on the x-axis
        xref2 = x.get()
        
        y = tk.IntVar() # Variable value for the x-axis data set given by Radiobutton
        y.set(0)        # Sets the default plot to show DM on the y-axis
        yref2 = y.get()

        datatypes1 = [ "DM", "Time", "StoN", "Width"]
        datatypes2 = [ "DM", "Time", "StoN", "Width"]
        
        # Method called by the Radiobuttons to check and update the xref and yref values if they have changed
        def rdbchange():
            global xref2
            global yref2
            
            if ((y.get() != yref2) or (x.get() != xref2)):
                xref2 = x.get()
                yref2 = y.get()
        
        # Radiobutton allowing the user to choose what data to display on the x-axis of the plot
        for val, datatypes1 in enumerate(datatypes1):
            tk.Radiobutton(self, 
                          text=datatypes1,
                          padx = 7, 
                          variable=x, 
                          value=val,
                          command = rdbchange,
                          font = fontx).grid(row = 5, column = 22 + val)
        
        # Radiobutton allowing the user to choose what data to display on the y-axis of the plot
        for val, datatypes2 in enumerate(datatypes2):
            tk.Radiobutton(self, 
                          text=datatypes2,
                          padx = 7, 
                          variable=y, 
                          value=val,
                          command = rdbchange,
                          font = fontx).grid(row = 6, column = 22 + val)
        
        # Button to display the plot defined by user's choice of x- and y-values
        self.show_button = tk.Button(self, text = "Show", command = lambda: self.canvasupdate(xref2, yref2))
        self.show_button.grid(row = 7, column = 23, columnspan = 2, padx = 15)
        
        # Button to view the previously classified plots
        self.view_button = tk.Button(self, text = "fdsfdsfds", command = lambda: self.controller.show_frame("main_frame"))
        self.view_button.config(height = 1, width = 20)
        self.view_button.grid(row = 21, column = 0, padx = (10,0))
        
        # Buttons to classify whether plot shows an FRB or not.
        self.frb_button = tk.Button(self,text = "Frb", command = lambda: self.frbchoice(odir_path,1))
        self.nfrb_button = tk.Button(self,text = "No frb", command = lambda: self.frbchoice(odir_path,0))
        self.frb_button.grid(row = 1, column = 21, padx = 15)
        self.nfrb_button.grid(row = 2, column = 21, padx = 15)
        
        # Buttons to move between plots
        self.right_btn = tk.Button(self, text = "Next plot", command = lambda: self.Right())
        self.left_btn = tk.Button(self, text = "Prev. plot", command = lambda: self.Left())
        self.right_btn.grid(row = 21, column = 8, pady = 5)
        self.left_btn.grid(row = 21, column = 7, pady = 5)

        
    def Right(self):
        global pics_ind2
        global pics
        global xref2
        global yref2
        
        if (pics_ind2 >= (len(pics) - 1)):  # If at the end of the array don't try to go further
            print("Reached end of files.")
        else:
            pics_ind2 += 1
            self.canvasupdate(xref2,yref2)
            
    def Left(self):
        global pics_ind2
        global pics
        global xref2
        global yref2
        
        if (pics_ind2 <= 0):   # If at the beginning on the array don't try to go further back
            print("At the first file already.")
        else:
            pics_ind2 -= 1
            self.canvasupdate(xref2,yref2)
     
    # Updates the progress bar    
    def bar(self):
        global length
        global progcount
        global progress
        
        progress['value']=progcount*100/length

    #Exits client
    def client_exit(self):
        exit()   
      
    #Method to put image in appropriate folder, make a note where that is, and change the image on display
    def frbchoice(self, folder, choice):
        global result_file
        global pics_ind2
        global pics
        global xref2
        global yref2
        global progcount
        
        progcount += 1
        self.bar()
        
        try:    #Only creates folders if they don't already exist
            os.mkdir(folder + "//frb")
            os.mkdir(folder + "//no_frb")
        except:
            pass
       
        #Adds path name and moves img to appropriate folder
        if (choice == 1):
            os.rename(pics[pics_ind2],folder + "\\frb\\" + pics[pics_ind2].split("\\")[-1])
            result_file.write(folder + "\\frb\\" + pics[pics_ind2].split("\\")[-1] + "\n")
        else:
            os.rename(pics[pics_ind2],folder + "\\no_frb\\" + pics[pics_ind2].split("\\")[-1])
            result_file.write(folder + "\\no_frb\\" + pics[pics_ind2].split("\\")[-1] + "\n")
  
        #Moves to next img
        if ((pics_ind >= (len(pics) - 1)) and (len(pics) != 1)):
            if (len(pics) != 0):
                pics_ind2 -= 1 
                self.canvasupdate(xref2,yref2)
                del pics[pics_ind2 + 1]
            print("Reached end of files.")
        elif (len(pics) == 1):
            self.canvas.delete("all")
        else:
            pics_ind2 += 1
            self.canvasupdate(xref2,yref2)
            del pics[pics_ind2 - 1]
            pics_ind2 -= 1 
            
    def canvasupdate(self, xref2, yref2):
        global pics_ind2
        self.canvas = FigureCanvasTkAgg(plotimg(pics[pics_ind2],xref2,yref2), self)
        self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 20, rowspan = 20)

#Returns a matplotlib plot to be displayed on the canvas         
def plotimg(path,xref,yref):
    
    #Imports data from the .dat file located at the specified path
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,4))
    Tfile.close()
    
    #Defines labels for the axes corresponding to the four different sets of data available
    axislabels = ["DM", "Time", "StoN", "Width"]
    columns = np.hsplit(c,4) #x- and yref values dm=0, time=1, ston=2, width=3
    
    #Creates the figure to be displayed. xref and yref corresponding to the chosen x and y values to be displayed
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(axislabels[xref])
    ax.set_ylabel(axislabels[yref])
    ax.scatter(columns[xref], columns[yref], s = 7)
    ax.set_xlim(left = 0) #Sets lower x-limit to zero
    
    return fig

 
def main():
    global pics_ind
    global pics
    global result_file
    global app
    global idir_path
    global odir_path
    global xref
    global yref
    global xref2
    global yref2
    global progcount
    global length
    global pics_ind2
    
    # Input directories as defined in relation to the work. dir.
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
    pics_ind2 = 0
    
    #Adds names of files ending with .gif in 'pics' list
    for file in glob.glob(idir_path + "/" + "*.dat"):
        pics.append(file)
    
    length = len(pics)
    progcount = 0
    
    app = start_app()
    app.mainloop()
   
    result_file.close()
   
if __name__ == '__main__':
    main()
