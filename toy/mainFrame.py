import tkinter as tk
import tkinter.ttk as ttk
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from globalDef import plotimg
from candFrame import candClass

fontx = ('Helvetica', 8)

class main_frame(tk.Frame):
    
    source_paths = []           # Paths to unclassified .dat files
    source_ind = 0              # Current position index used in source_paths array
    odir_path = ""              # Path to the output directory
    progcount = 0               # Counts how many files that have been classified in the session
    length = 0                  # Number of initially unclassified .dat files  
    result_file = None          # File containing paths to classified .dat files
    xref = 0                    # Indicates which axis is being used in the x-direction
    yref = 0                    # Indicates which axis is being used in the y-direction
    
    # Initialises the frame here 
    def __init__(self, master, controller):
       
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
        #menu.add_cascade(label = "View", menu = editmenu)
        
    # Method defining the Labels (including the canvas) part of the GUI
    def labels(self):

        self.canvasupdate(self.xref, self.yref)   # Plots the first .dat file in the directory as default
        tk.Label(self, text = "Y:", font = "Helvetica 9 bold").grid(row = 11, column = 54)  # Y-label for y axis radiobuttons
        tk.Label(self, text = "X:", font = "Helvetica 9 bold").grid(row = 12, column = 54)  # X-label for x axis radiobuttons
        
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
    
    # Method defining the buttons available on the GUI
    def buttons(self):
        
        x = tk.IntVar() # Variable value for the x-axis data set given by Radiobutton
        x.set(1)        # Sets the default plot to show time on the x-axis
        main_frame.xref = x.get()
        y = tk.IntVar() # Variable value for the x-axis data set given by Radiobutton
        y.set(0)        # Sets the default plot to show DM on the y-axis
        main_frame.yref = y.get()

        datatypes1 = [ "DM", "Time", "s/n", "Width"]
        datatypes2 = [ "DM", "Time", "s/n", "Width"]
        
        # Method called by the Radiobuttons to check and update the xref and yref values if they have changed
        def rdbchange():
            
            # Only updates xref and yref if one or both have changed
            if ((y.get() != main_frame.yref) or (x.get() != main_frame.xref)):
                main_frame.xref = x.get()
                main_frame.yref = y.get()
        
        # Radiobutton allowing the user to choose what data to display on the x-axis of the plot
        for val, datatypes1 in enumerate(datatypes1):
            tk.Radiobutton(self, 
                          text=datatypes1,
                          padx = 7, 
                          variable=x, 
                          value=val,
                          command = rdbchange,
                          font = fontx).grid(row = 12, column = 56 + val)
        
        # Radiobutton allowing the user to choose what data to display on the y-axis of the plot
        for val, datatypes2 in enumerate(datatypes2):
            tk.Radiobutton(self, 
                          text=datatypes2,
                          padx = 7, 
                          variable=y, 
                          value=val,
                          command = rdbchange,
                          font = fontx).grid(row = 11, column = 56 + val)
        
        # Button to display the plot defined by user's choice of x- and y-values
        self.show_button = tk.Button(self, text = "Show", command = lambda: self.canvasupdate(main_frame.xref, main_frame.yref))
        self.show_button.grid(row = 13, column = 55, columnspan = 5, rowspan = 4)
        
        # Button to view the previously classified plots
        self.prev_button = tk.Button(self, text = "View plot previews", command = lambda: self.controller.show_frame("preview_frame"))
        self.prev_button.config(height = 1, width = 20)
        self.prev_button.grid(row = 51, column = 59, columnspan = 10, padx = (10,5), pady = (5,5))
        
        # Button to view the previously classified plots
        self.view_button = tk.Button(self, text = "View classified plots", command = lambda: self.controller.show_frame("candClass"))
        self.view_button.config(height = 1, width = 20)
        self.view_button.grid(row = 51, column = 1, columnspan = 10, padx = (10,0), pady = (5,5))
        
        # Buttons to classify whether plot shows an FRB or not.
        self.frb_button = tk.Button(self,text = "Frb", command = lambda: self.frbchoice(main_frame.odir_path,1))
        self.nfrb_button = tk.Button(self,text = "No frb", command = lambda: self.frbchoice(main_frame.odir_path,0))
        self.frb_button.config(height = 2, width = 6)
        self.nfrb_button.config(height = 2, width = 6)
        self.frb_button.grid(row = 2, column = 51, rowspan = 3, columnspan = 5, padx = 15)
        self.nfrb_button.grid(row = 6, column = 51, rowspan = 3, columnspan = 5, padx = 15)
        
        # Buttons to move between plots
        self.right_btn = tk.Button(self, text = "Next plot", command = lambda: self.Right())
        self.left_btn = tk.Button(self, text = "Prev. plot", command = lambda: self.Left())
        self.right_btn.grid(row = 50, column = 24, columnspan = 6, rowspan = 3, pady = (5,5), sticky = "nw")
        self.left_btn.grid(row = 50, column = 17, columnspan = 6, rowspan = 3, pady = (5,5), sticky = "nw")
        
    # Method to display the next image in the path space
    def Right(self):
        if (self.source_ind >= (len(self.source_paths) - 1)):  # If at the end of the array don't try to go further
            print("Reached end of files.")
        else:   # Increase the index by 1 and update the canvas accordingly
            self.source_ind += 1
            self.canvasupdate(self.xref,self.yref)
    
    # Method to display the previous image in the path space      
    def Left(self):
        if (self.source_ind <= 0):   # If at the beginning on the array don't try to go further back
            print("At the first file already.")
        else:       # Decrease the index by 1 and update the canvas accordingly
            self.source_ind -= 1
            self.canvasupdate(self.xref,self.yref)
        
    #Changing image with left or right key
    def key1(self, event):

        if event.keysym == 'Right':     # Right arrow pressed
            if (self.source_ind >= (len(self.source_paths) - 1)):       # If at the end of the array don't try to go further
                print("Reached end of files.")
            else:   # Increase the index by 1 and update the canvas accordingly
                self.source_ind += 1
                self.canvasupdate(self.xref,self.yref)
        elif event.keysym == 'Left':    # Left arrow pressed
            if (self.source_ind <= 0):                     # If at the beginning on the array don't try to go further back
                print("At the first file already.")
            else:   # Decrease the index by 1 and update the canvas accordingly
                self.source_ind -= 1
                self.canvasupdate(self.xref,self.yref)
     
    # Updates the progress bar    
    def bar(self):
        self.progress['value']=main_frame.progcount*100/main_frame.length

    #Exits client
    def client_exit(self):
        exit()   
      
    #Method to put image in appropriate folder, make a note where that is, and change the image on display
    def frbchoice(self, folder, choice):
        main_frame.progcount += 1
        self.bar()
        
        try:    #Only creates folders if they don't already exist
            os.mkdir(folder + "//frb")
            os.mkdir(folder + "//no_frb")
        except:
            pass
       
        #Adds path name and moves img to appropriate folder
        if (choice == 1):
            os.rename(self.source_paths[self.source_ind],folder + "\\frb\\" + self.source_paths[self.source_ind].split("\\")[-1])
            self.result_file.write(folder + "\\frb\\" + self.source_paths[self.source_ind].split("\\")[-1] + "\n")
            candClass.frb_paths.append(folder + "\\frb\\" + self.source_paths[self.source_ind].split("\\")[-1])
        else:
            os.rename(self.source_paths[self.source_ind],folder + "\\no_frb\\" + self.source_paths[self.source_ind].split("\\")[-1])
            self.result_file.write(folder + "\\no_frb\\" + self.source_paths[self.source_ind].split("\\")[-1] + "\n")
            candClass.nfrb_paths.append(folder + "\\no_frb\\" + self.source_paths[self.source_ind].split("\\")[-1])
            
  
        #Moves to next img
        # Goes in here if at the end of the files
        if ((self.source_ind >= (len(self.source_paths) - 1)) and (len(self.source_paths) != 1)):
            if (len(self.source_paths) != 0):       # If there are unclassified .dat files remaining then show previous files
                self.source_ind -= 1 
                self.canvasupdate(self.xref,self.yref)
                del self.source_paths[self.source_ind + 1]
            print("Reached end of files.")
        elif (len(self.source_paths) == 1):         # Goes in here if this file was the last one, displays empty canvas
            self.canvas.delete("all")
        else:                                       # Goes in here if there are more .dat files after the current one, displays it
            self.source_ind += 1
            self.canvasupdate(self.xref,self.yref)
            del main_frame.self.source_paths[self.source_ind - 1]
            self.source_ind -= 1 
            
    # Method to initialise a plot on the canvas
    def canvasupdate(self, xref, yref):
        self.canvas = FigureCanvasTkAgg(plotimg(self.source_paths[self.source_ind],xref,yref), self)
        self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)