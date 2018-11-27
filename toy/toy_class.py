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
        for F in (main_frame, frame1, preview_frame):
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
        
        menubar = frame.menuobj()
        self.configure(menu=menubar)
     
     
        
class main_frame(tk.Frame):
    
    def __init__(self, master, controller):
        global source_paths
        global source_ind
       
        #MASTER FRAME THINGS HERE
        tk.Frame.__init__(self, master)
        self.controller = controller            
        self.controller.title("FRB viewer")
        self.grid(row=0,column=0)
        self.controller.configure(background = 'white')
        self.controller.bind('<Key>', lambda event: self.key1(event))
        
        self.buttons()
        self.menuobj()
        self.labels()
        self.textbox()
            
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
        
        editmenu = tk.Menu(menu)
        editmenu.add_command(label = "Choose output directory...",command = self.client_exit)
        menu.add_cascade(label = "View", menu = editmenu)
        
    # Method defining the Labels (including the canvas) part of the GUI
    def labels(self):
        global xref
        global yref
        global progress

        self.canvasupdate(xref, yref)   #Shows the first plot of the data set as default at start-up
        tk.Label(self, text = "X:", font = "Helvetica 9 bold").grid(row = 11, column = 54)
        tk.Label(self, text = "Y:", font = "Helvetica 9 bold").grid(row = 12, column = 54)
        
        # Progress bar displaying fraction of classified files
        progress = ttk.Progressbar(self, orient="horizontal",length=100,mode='determinate')
        progress.grid(row = 51, column = 40, sticky = "nw", columnspan = 10, rowspan = 2, pady = (5,0))
        
    # Method defining the textbox part of the GUI
    def textbox(self):
        
        self.T = tk.Text(self,height = 6,width = 35)
        self.S = tk.Scrollbar(self)
        self.S.config(command = self.T.yview)
        self.T.config(yscrollcommand = self.S.set)
        self.T.grid(row = 1,column = 56, rowspan = 10, columnspan = 40, sticky = "nw", padx = (0, 15))
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
                          font = fontx).grid(row = 11, column = 56 + val)
        
        # Radiobutton allowing the user to choose what data to display on the y-axis of the plot
        for val, datatypes2 in enumerate(datatypes2):
            tk.Radiobutton(self, 
                          text=datatypes2,
                          padx = 7, 
                          variable=y, 
                          value=val,
                          command = rdbchange,
                          font = fontx).grid(row = 12, column = 56 + val)
        
        # Button to display the plot defined by user's choice of x- and y-values
        self.show_button = tk.Button(self, text = "Show", command = lambda: self.canvasupdate(xref, yref))
        self.show_button.grid(row = 13, column = 55, columnspan = 5, rowspan = 4)
        
        # Button to view the previously classified plots
        self.prev_button = tk.Button(self, text = "View plot previews", command = lambda: self.controller.show_frame("preview_frame"))
        self.prev_button.config(height = 1, width = 20)
        self.prev_button.grid(row = 51, column = 59, columnspan = 10, padx = (10,5), pady = (5,5))
        
        # Button to view the previously classified plots
        self.view_button = tk.Button(self, text = "View classified plots", command = lambda: self.controller.show_frame("frame1"))
        self.view_button.config(height = 1, width = 20)
        self.view_button.grid(row = 51, column = 1, columnspan = 10, padx = (10,0), pady = (5,5))
        
        # Buttons to classify whether plot shows an FRB or not.
        self.frb_button = tk.Button(self,text = "Frb", command = lambda: self.frbchoice(odir_path,1))
        self.nfrb_button = tk.Button(self,text = "No frb", command = lambda: self.frbchoice(odir_path,0))
        self.frb_button.config(height = 2, width = 6)
        self.nfrb_button.config(height = 2, width = 6)
        self.frb_button.grid(row = 2, column = 51, rowspan = 3, columnspan = 5, padx = 15)
        self.nfrb_button.grid(row = 6, column = 51, rowspan = 3, columnspan = 5, padx = 15)
        
        # Buttons to move between plots
        self.right_btn = tk.Button(self, text = "Next plot", command = lambda: self.Right())
        self.left_btn = tk.Button(self, text = "Prev. plot", command = lambda: self.Left())
        self.right_btn.grid(row = 50, column = 24, columnspan = 6, rowspan = 3, pady = (5,5), sticky = "nw")
        self.left_btn.grid(row = 50, column = 17, columnspan = 6, rowspan = 3, pady = (5,5), sticky = "nw")
        
    def Right(self):
        global source_ind
        global source_paths
        global xref
        global yref
        
        if (source_ind >= (len(source_paths) - 1)):  # If at the end of the array don't try to go further
            print("Reached end of files.")
        else:
            source_ind += 1
            self.canvasupdate(xref,yref)
            
    def Left(self):
        global source_ind
        global source_paths
        global xref
        global yref
        
        if (source_ind <= 0):   # If at the beginning on the array don't try to go further back
            print("At the first file already.")
        else:
            source_ind -= 1
            self.canvasupdate(xref,yref)
        
    #Changing image with left or right key
    def key1(self, event):
        global source_ind
        global source_paths
        global xref
        global yref
        
        print("besh")
        
        if event.keysym == 'Right':
            if (source_ind >= (len(source_paths) - 1)):       # If at the end of the array don't try to go further
                print("Reached end of files.")
            else:
                source_ind += 1
                self.canvasupdate(xref,yref)
        elif event.keysym == 'Left':
            if (source_ind <= 0):                     # If at the beginning on the array don't try to go further back
                print("At the first file already.")
            else:
                source_ind -= 1
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
        global source_ind
        global source_paths
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
            os.rename(source_paths[source_ind],folder + "\\frb\\" + source_paths[source_ind].split("\\")[-1])
            result_file.write(folder + "\\frb\\" + source_paths[source_ind].split("\\")[-1] + "\n")
        else:
            os.rename(source_paths[source_ind],folder + "\\no_frb\\" + source_paths[source_ind].split("\\")[-1])
            result_file.write(folder + "\\no_frb\\" + source_paths[source_ind].split("\\")[-1] + "\n")
  
        #Moves to next img
        if ((source_ind >= (len(source_paths) - 1)) and (len(source_paths) != 1)):
            if (len(source_paths) != 0):
                source_ind -= 1 
                self.canvasupdate(xref,yref)
                del source_paths[source_ind + 1]
            print("Reached end of files.")
        elif (len(source_paths) == 1):
            self.canvas.delete("all")
        else:
            source_ind += 1
            self.canvasupdate(xref,yref)
            del source_paths[source_ind - 1]
            source_ind -= 1 
            
    def canvasupdate(self, xref, yref):
        self.canvas = FigureCanvasTkAgg(plotimg(source_paths[source_ind],xref,yref), self)
        self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)


class frame1(tk.Frame):
    
    def __init__(self, master, controller):
        global source_paths
        global review_ind
       
        #MASTER FRAME THINGS HERE
        tk.Frame.__init__(self, master)
        self.controller = controller            
        self.controller.title("FRB viewer")
        self.grid(row=0,column=0)
        self.controller.configure(background = 'black')
        
        rows = 0
        while rows < 100:
            self.controller.rowconfigure(rows, weight=1)
            self.controller.columnconfigure(rows,weight=1)
            rows += 1
        
        self.buttons()
        self.menuobj()
        self.labels()
        self.textbox()
    
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
        
        editmenu = tk.Menu(menu)
        editmenu.add_command(label = "Choose output directory...",command = self.client_exit)
        menu.add_cascade(label = "View", menu = editmenu)
        
    # Method defining the Labels (including the canvas) part of the GUI
    def labels(self):
        global xref2
        global yref2
        global progress
        
        #self.canvasupdate(xref2, yref2)   #Shows the first plot of the data set as default at start-up
        self.emptycanvas()
        tk.Label(self, text = "X:", font = "Helvetica 9 bold").grid(row = 11, column = 54)
        tk.Label(self, text = "Y:", font = "Helvetica 9 bold").grid(row = 12, column = 54)
        
    # Method defining the textbox part of the GUI
    def textbox(self):
        
        self.T = tk.Text(self,height = 6,width = 35)
        self.S = tk.Scrollbar(self)
        self.S.config(command = self.T.yview)
        self.T.config(yscrollcommand = self.S.set)
        self.T.grid(row = 1,column = 56, rowspan = 10, columnspan = 40, sticky = "nw", padx = (0, 15))
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
                          font = fontx).grid(row = 11, column = 56 + val)
        
        # Radiobutton allowing the user to choose what data to display on the y-axis of the plot
        for val, datatypes2 in enumerate(datatypes2):
            tk.Radiobutton(self, 
                          text=datatypes2,
                          padx = 7, 
                          variable=y, 
                          value=val,
                          command = rdbchange,
                          font = fontx).grid(row = 12, column = 56 + val)
        
        # Button to display the plot defined by user's choice of x- and y-values
        self.show_button = tk.Button(self, text = "Show", command = lambda: self.canvasupdate(xref2, yref2))
        self.show_button.grid(row = 13, column = 55, columnspan = 5, rowspan = 4)
        
        # Button to view the previously classified plots
        self.view_button = tk.Button(self, text = "fdsfdsfds", command = lambda: self.controller.show_frame("main_frame"))
        self.view_button.config(height = 1, width = 20)
        self.view_button.grid(row = 51, column = 1, columnspan = 10, padx = (10,0), pady = (5,5))
        
        # Buttons to classify whether plot shows an FRB or not.
        self.frb_button = tk.Button(self,text = "Frb", command = lambda: self.display_choice(1))
        self.nfrb_button = tk.Button(self,text = "No frb", command = lambda: self.display_choice(0))
        self.frb_button.config(height = 2, width = 6)
        self.nfrb_button.config(height = 2, width = 6)
        self.frb_button.grid(row = 2, column = 51, rowspan = 3, columnspan = 5, padx = 15)
        self.nfrb_button.grid(row = 6, column = 51, rowspan = 3, columnspan = 5, padx = 15)
        
        # Buttons to move between plots
        self.right_btn = tk.Button(self, text = "Next plot", command = lambda: self.Right())
        self.left_btn = tk.Button(self, text = "Prev. plot", command = lambda: self.Left())
        self.right_btn.grid(row = 50, column = 24, columnspan = 6, rowspan = 3, sticky = "nw", pady = (5,5))
        self.left_btn.grid(row = 50, column = 17, columnspan = 6, rowspan = 3, sticky = "nw", pady = (5,5))

        
    def Right(self):
        global review_ind
        global current_choice
        global xref2
        global yref2
        
        if (review_ind >= (len(current_choice) - 1)):  # If at the end of the array don't try to go further
            print("Reached end of files.")
        else:
            review_ind += 1
            self.canvasupdate(current_choice,xref2,yref2)
            
    def Left(self):
        global review_ind
        global current_choice
        global xref2
        global yref2
        
        if (review_ind <= 0):   # If at the beginning on the array don't try to go further back
            print("At the first file already.")
        else:
            review_ind -= 1
            self.canvasupdate(current_choice,xref2,yref2)

    #Exits client
    def client_exit(self):
        exit()   
      
    #Method to put image in appropriate folder, make a note where that is, and change the image on display
    def display_choice(self, choice):
        global current_choice
        global xref2
        global yref2
        global review_ind
        global frb_paths
        global nfrb_paths
        
        review_ind = 0
        #current_choice = folder
        
        if (choice == 1):
            self.frb_button.config(relief="sunken")
            self.nfrb_button.config(relief="raised")
            current_choice = frb_paths
            self.canvasupdate(current_choice,xref2,yref2)
        else:
            self.nfrb_button.config(relief="sunken")
            self.frb_button.config(relief="raised")
            current_choice = nfrb_paths
            self.canvasupdate(current_choice,xref2,yref2)
        
    def canvasupdate(self, folder, xref2, yref2):
        global review_ind
        self.canvas = FigureCanvasTkAgg(plotimg(folder[review_ind],xref2,yref2), self)
        self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
        
    def emptycanvas(self):
    
        fig = Figure(figsize = (8,6))
        ax = fig.add_subplot(111)
        ax.set_xlabel("Empty")
        ax.set_ylabel("Empty")
        ax.set_xlim(left = 0) #Sets lower x-limit to zero
        self.canvas = FigureCanvasTkAgg(fig, self)
        self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
    
 


#Frame to display thumbnail images of plots for the user to choose from
class preview_frame(tk.Frame):
    
    def __init__(self, master, controller):
        global source_paths
        global review_ind
       
        #MASTER FRAME THINGS HERE
        tk.Frame.__init__(self, master)
        self.controller = controller            
        self.controller.title("FRB viewer")
        self.grid(row=0,column=0)
        self.controller.configure(background = 'black')
        
        rows = 0
        while rows < 100:
            self.controller.rowconfigure(rows, weight=1)
            self.controller.columnconfigure(rows,weight=1)
            rows += 1
        
        self.buttons()
        self.menuobj()
        self.labels()
    
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
        
        editmenu = tk.Menu(menu)
        editmenu.add_command(label = "Choose output directory...",command = self.client_exit)
        menu.add_cascade(label = "Viewfff", menu = editmenu)
        
    # Method defining the Labels (including the canvas) part of the GUI
    def labels(self):
        global xref2
        global yref2
        global progress
        
        #self.canvasupdate(xref2, yref2)   #Shows the first plot of the data set as default at start-up
        self.subplot_canv(xref,yref)
        tk.Label(self, text = "X:", font = "Helvetica 9 bold").grid(row = 7, column = 54)
        tk.Label(self, text = "Y:", font = "Helvetica 9 bold").grid(row = 8, column = 54)
        
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
                          font = fontx).grid(row = 7, column = 58 + val)
        
        # Radiobutton allowing the user to choose what data to display on the y-axis of the plot
        for val, datatypes2 in enumerate(datatypes2):
            tk.Radiobutton(self, 
                          text=datatypes2,
                          padx = 7, 
                          variable=y, 
                          value=val,
                          command = rdbchange,
                          font = fontx).grid(row = 8, column = 58 + val)
        
        # Button to display the plot defined by user's choice of x- and y-values
        self.show_button = tk.Button(self, text = "Show", command = lambda: self.subplot_canv(xref,yref))
        self.show_button.grid(row = 8, column = 56, columnspan = 5, rowspan = 4)
        
        # Button to view the previously classified plots
        self.view_button = tk.Button(self, text = "Return", command = lambda: self.controller.show_frame("main_frame"))
        self.view_button.config(height = 1, width = 20)
        self.view_button.grid(row = 51, column = 0, columnspan = 10, padx = (5,0), pady = (5,5))
        
        # Buttons to move between plots
        self.right_btn = tk.Button(self, text = "Next plot", command = lambda: self.Right())
        self.left_btn = tk.Button(self, text = "Prev. plot", command = lambda: self.Left())
        self.right_btn.grid(row = 50, column = 14, columnspan = 6, rowspan = 3, sticky = "nw", pady = (5,5))
        self.left_btn.grid(row = 50, column = 11, columnspan = 6, rowspan = 3, sticky = "nw", pady = (5,5))

    # Shows the next 4 thumbnail plots   
    def Right(self):
        global preview_ind
        global source_paths
        global xref
        global yref
        
        if (preview_ind >= (len(source_paths) - 4)):    # If at the end of the array don't try to go further
            print("Reached end of files.")
        elif (preview_ind >= (len(source_paths) - 8)):  # If less than 4 plots remaining draw the last 4 plots to avoid error
            preview_ind = (len(source_paths) - 4)
            self.subplot_canv(xref,yref)
        else:                                           # Move the index forward 4 steps to display next 4 plots
            preview_ind += 4
            self.subplot_canv(xref,yref)
       
    # Shows the previous 4 thumbnail plots
    def Left(self):
        global preview_ind
        global source_paths
        global xref
        global yref
        
        if (preview_ind <= 0):                          # If at the beginning on the array don't try to go further back
            print("At the first file already.")
        elif (preview_ind <= 4):                        # If less than 4 plots remaining draw the first 4 plots to avoid error
            preview_ind = 0
            self.subplot_canv(xref,yref)
        else:                                           # Move the index back 4 steps to display previous 4 plots
            preview_ind -= 4
            self.subplot_canv(xref,yref)

    #Exits client
    def client_exit(self):
        exit()
    
    # Canvas function to display the subplots in the window
    def subplot_canv(self,xref,yref):
        global source_paths
        global preview_ind
        
        fig = Figure(figsize = (8,6))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        fig.tight_layout()
        
        # Draws next 4 plots in the subplots
        draw_subplot(source_paths[preview_ind + 0],xref,yref,ax1,fig)
        draw_subplot(source_paths[preview_ind + 1],xref,yref,ax2,fig)
        draw_subplot(source_paths[preview_ind + 2],xref,yref,ax3,fig)
        draw_subplot(source_paths[preview_ind + 3],xref,yref,ax4,fig)
        
        fig.subplots_adjust(left=0.1, bottom=0.1, right=None, top=None, wspace=0.4, hspace=0.4) # Defines subplot space allocation
        self.canvas1 = FigureCanvasTkAgg(fig, self)
        self.canvas1.get_tk_widget().grid(row = 0, column = 0, columnspan = 25, sticky = "nw", rowspan = 25)
        
#Returns a matplotlib subplot to be displayed       
def draw_subplot(path,xref,yref,ax,fig):
    
    #Imports data from the .dat file located at the specified path
    Tfile = open(path,'r')
    data = np.fromfile(Tfile,np.float32,-1) 
    c = data.reshape((-1,4))
    Tfile.close()
    
    #Defines labels for the axes corresponding to the four different sets of data available
    axislabels = ["DM", "Time", "StoN", "Width"]
    columns = np.hsplit(c,4) #x- and yref values dm=0, time=1, ston=2, width=3
    
    #Creates the figure to be displayed. xref and yref corresponding to the chosen x and y values to be displayed
    ax.set_xlabel(axislabels[xref])
    ax.set_ylabel(axislabels[yref])
    ax.scatter(columns[xref], columns[yref], s = 7)
    ax.set_xlim(left = 0) #Sets lower x-limit to zero
    
    return fig
           
    
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
    fig = Figure(figsize = (8,6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(axislabels[xref])
    ax.set_ylabel(axislabels[yref])
    ax.scatter(columns[xref], columns[yref], s = 10, facecolors='none', edgecolors='b')
    ax.set_xlim(left = 0) #Sets lower x-limit to zero
    ax.set_title(path)
    return fig

 
def main():
    global source_ind
    global source_paths
    global frb_paths
    global nfrb_paths
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
    global review_ind
    global preview_ind

    
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
   
    source_paths = []       #List of filenames to be viewed in program
    frb_paths = []
    nfrb_paths = []

    source_ind = 0    #Index for the img currently being viewed
    review_ind = 0
    preview_ind = 0
    
    #Adds names of files ending with .gif in 'source_paths' list
    for file in glob.glob(idir_path + "/" + "*.dat"):
        source_paths.append(file)
        
    for file in glob.glob(odir_path + "\\frb\\" + "*.dat"):
        frb_paths.append(file)
        
    for file in glob.glob(odir_path + "\\no_frb\\" + "*.dat"):
        nfrb_paths.append(file)
    
    length = len(source_paths)
    progcount = 0

    app = start_app()
    app.mainloop()
   
    result_file.close()
   
if __name__ == '__main__':
    main()
