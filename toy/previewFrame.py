import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from globalDef import draw_subplot

fontx = ('Helvetica', 8)

#Frame to display thumbnail images of plots for the user to choose from
class preview_frame(tk.Frame):
    
    source_paths = []
    preview_ind = 0
    
    def __init__(self, master, controller):
       
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

        datatypes1 = [ "DM", "Time", "s/n", "Width"]
        datatypes2 = [ "DM", "Time", "s/n", "Width"]
        
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
        
        # Button to display or hide classified plots
        self.class_btn = tk.Button(self, text = "Show classifications", command = lambda: self.Right())
        self.class_btn.grid(row = 10, column = 56, columnspan = 6, rowspan = 3, sticky = "nw", pady = (5,5))
        
    # Shows the next 4 thumbnail plots   
    def Right(self):
        source_paths = preview_frame.source_paths
        preview_ind = preview_frame.preview_ind
        global xref
        global yref
        
        if (preview_ind >= (len(source_paths) - 4)):    # If at the end of the array don't try to go further
            print("Reached end of files.")
        elif (preview_ind >= (len(source_paths) - 8)):  # If less than 4 plots remaining draw the last 4 plots to avoid error
            preview_frame.preview_ind = (len(source_paths) - 4)
            self.subplot_canv(xref,yref)
        else:                                           # Move the index forward 4 steps to display next 4 plots
            preview_frame.preview_ind += 4
            self.subplot_canv(xref,yref)
       
    # Shows the previous 4 thumbnail plots
    def Left(self):
        preview_ind = preview_frame.preview_ind
        global xref
        global yref
        
        if (preview_ind <= 0):                          # If at the beginning on the array don't try to go further back
            print("At the first file already.")
        elif (preview_ind <= 4):                        # If less than 4 plots remaining draw the first 4 plots to avoid error
            preview_frame.preview_ind = 0
            self.subplot_canv(xref,yref)
        else:                                           # Move the index back 4 steps to display previous 4 plots
            preview_frame.preview_ind -= 4
            self.subplot_canv(xref,yref)
        
    #Exits client
    def client_exit(self):
        exit()
    
    # Canvas function to display the subplots in the window
    def subplot_canv(self,xref,yref):
        source_paths = preview_frame.source_paths
        preview_ind = preview_frame.preview_ind
        
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
        
        
        