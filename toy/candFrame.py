import tkinter as tk
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from globalDef import candPlot

fontx = ('Helvetica', 8)
class_colours = ["b", "m", "r"]

class candClass(tk.Frame):
    
    review_ind_frb = 0
    review_ind_nfrb = 0
    choice = 0
    current_choice = []
    xref = 0
    yref = 0
    frb_paths = []
    nfrb_paths = []
    excellent = []
    good = []
    least_acc = []
    
    axes = []
    candAx = None
    candFig = None

    
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
        #self.canvasPlot(xref2, yref2)   #Shows the first plot of the data set as default at start-up
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
    
    # Method defining the buttons available on the GUI
    def buttons(self):
        
        x = tk.IntVar() # Variable value for the x-axis data set given by Radiobutton
        x.set(1)        # Sets the default plot to show time on the x-axis
        candClass.xref = x.get()
        
        y = tk.IntVar() # Variable value for the x-axis data set given by Radiobutton
        y.set(0)        # Sets the default plot to show DM on the y-axis
        candClass.yref = y.get()

        datatypes1 = [ "DM", "Time", "s/n", "Width"]
        datatypes2 = [ "DM", "Time", "s/n", "Width"]
        
        # Method called by the Radiobuttons to check and update the xref and yref values if they have changed
        def rdbchange():
            
            if ((y.get() != candClass.yref) or (x.get() != candClass.xref)):
                candClass.xref = x.get()
                candClass.yref = y.get()
        
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
        self.show_button = tk.Button(self, text = "Show", command = lambda: self.canvasPlot(candClass.xref, candClass.yref))
        self.show_button.grid(row = 13, column = 55, columnspan = 5, rowspan = 4)
        
        # Button to view the previously classified plots
        self.view_button = tk.Button(self, text = "Return", command = lambda: self.controller.show_frame("main_frame"))
        self.view_button.config(height = 1, width = 20)
        self.view_button.grid(row = 51, column = 1, columnspan = 10, padx = (10,0), pady = (5,5))
        
        # Buttons to classify whether plot shows an FRB or not.
        self.frb_button = tk.Button(self,text = "Frb", command = lambda: self.display_choice(1))
        self.nfrb_button = tk.Button(self,text = "No frb", command = lambda: self.eraseCand())
        self.frb_button.config(height = 2, width = 6)
        self.nfrb_button.config(height = 2, width = 6)
        self.frb_button.grid(row = 2, column = 51, rowspan = 3, columnspan = 5, padx = 15)
        self.nfrb_button.grid(row = 6, column = 51, rowspan = 3, columnspan = 5, padx = 15)
        
        # Buttons to move between plots
        self.right_btn = tk.Button(self, text = "Next plot", command = lambda: self.Right())
        self.left_btn = tk.Button(self, text = "Prev. plot", command = lambda: self.Left())
        self.right_btn.grid(row = 50, column = 24, columnspan = 6, rowspan = 3, sticky = "nw", pady = (5,5))
        self.left_btn.grid(row = 50, column = 17, columnspan = 6, rowspan = 3, sticky = "nw", pady = (5,5))
        
        # Button to display or hide classified plots
        self.class_btn = tk.Button(self, text = "Show classifications", command = lambda: self.candData(self.xref,self.yref))
        self.class_btn.config(height = 2, width = 20)
        self.class_btn.grid(row = 18, column = 51, columnspan = 6, rowspan = 4, pady = (5,5), padx = (15,0))

        
    def Right(self):
        if candClass.choice == 1:
            review_ind = candClass.review_ind_frb
        else:
            review_ind = candClass.review_ind_nfrb
            
        current_choice = candClass.current_choice
        xref = candClass.xref
        yref = candClass.yref
        
        if (review_ind >= (len(current_choice) - 1)):  # If at the end of the array don't try to go further
            print("Reached end of files.")
        else:
            if candClass.choice == 1:
                candClass.review_ind_frb += 1
            else:
                candClass.review_ind_nfrb += 1
            self.canvasPlot(current_choice,xref,yref)
        
    def Left(self):
        if candClass.choice == 1:
            review_ind = candClass.review_ind_frb
        else:
            review_ind = candClass.review_ind_nfrb

        current_choice = candClass.current_choice
        xref = candClass.xref
        yref = candClass.yref
        
        if (review_ind <= 0):   # If at the beginning on the array don't try to go further back
            print("At the first file already.")
        else:
            if candClass.choice == 1:
                candClass.review_ind_frb -= 1
            else:
                candClass.review_ind_nfrb -= 1
            self.canvasPlot(current_choice,xref,yref)

    #Exits client
    def client_exit(self):
        exit()   
      
    #Method to put image in appropriate folder, make a note where that is, and change the image on display
    def display_choice(self, xchoice):
        xref = candClass.xref
        yref = candClass.yref
        candClass.choice = xchoice
        
        if (xchoice == 1):
            self.frb_button.config(relief="sunken")
            self.nfrb_button.config(relief="raised")
            candClass.current_choice = candClass.frb_paths
            self.canvasPlot(candClass.current_choice,xref,yref)
            
        else:
            self.nfrb_button.config(relief="sunken")
            self.frb_button.config(relief="raised")
            candClass.current_choice = candClass.nfrb_paths
            self.canvasPlot(candClass.current_choice,xref,yref)
        
    def canvasPlot(self, folder, xref, yref):
        if candClass.choice == 1:
            review_ind = self.review_ind_frb
        else:
            review_ind = self.review_ind_nfrb
        
        temp = candPlot(folder[review_ind],xref,yref)
        candClass.candFigNC = temp[0]
        candClass.candFigC = temp[1]
        self.drawGray(xref, yref)
        self.canvas = FigureCanvasTkAgg(self.candFig, self)
        #candPlot(folder[review_ind],xref,yref)
        self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
    
    def candData(self, xref, yref):
        candFile = self.current_choice[self.review_ind_frb].replace('.dat','_c.csv')
        dataset = pd.read_csv(candFile)
    
        X = dataset.iloc[:,1:5].values
        X = np.array(X)
        
        self.excellent = []
        self.good = []
        self.least_acc = []
        
        for i in range(len(X[:,3])):
            if X[:,3][i] == 3:
                self.excellent.append(X[i])
            elif X[:,3][i] == 2:
                self.good.append(X[i])
            elif X[:,3][i] == 1:
                self.least_acc.append(X[i])
        self.excellent = np.array(self.excellent)
        self.good = np.array(self.good)
        self.least_acc = np.array(self.least_acc)

        if len(self.excellent) > 0:
            self.axes.append(self.candAx.scatter(self.excellent[:,xref], self.excellent[:,yref], color = "r", alpha = 1, vmin = -1, s = 6, label = "Excellent"))
        if len(self.good) > 0:
            self.axes.append(self.candAx.scatter(self.good[:,xref], self.good[:,yref], color = "m", alpha = 1, vmin = -1, s = 6, label = "Good"))
        if len(self.least_acc) > 0:
            self.axes.append(self.candAx.scatter(self.least_acc[:,xref], self.least_acc[:,yref], color = "b", alpha = 1, vmin = -1, s = 6, label = "Least acceptable"))
        
        self.canvas = FigureCanvasTkAgg(self.candFig, self)
        self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
        
    def eraseCand(self):
        if len(self.axes) > 0:
            for i in self.axes:
                i.remove()
        else:
            print("No candidates highlighted.")

        self.canvas = FigureCanvasTkAgg(self.candFig, self)
        self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
        self.axes = []
    def drawGray(self, xref, yref):
        
        if candClass.choice == 1:
            review_ind = self.review_ind_frb
        else:
            review_ind = self.review_ind_nfrb
        
        path = self.current_choice[review_ind]
        #Imports data from the .dat file located at the specified path
        file = open(path,'r')
        data = np.fromfile(file,np.float32,-1) 
        c = data.reshape((-1,4))
        file.close()
    
        #Defines labels for the axes corresponding to the four different sets of data available
        axislabels = ["DM", "Time", "s/n", "Width"]
        columns = np.hsplit(c,4) #x- and yref values dm=0, time=1, s/n=2, width=3
        
        #Creates the figure to be displayed. xref and yref corresponding to the chosen x and y values to be displayed
        self.candFig = Figure(figsize = (8,6))
        self.candAx = self.candFig.add_subplot(111)
        self.candAx.set_xlabel(axislabels[xref])
        self.candAx.set_ylabel(axislabels[yref])
        self.candAx.scatter(columns[xref], columns[yref], color = "0.7", alpha = 1, vmin = -1, s = 6, label = "RFI/Background")
        self.candAx.set_xlim(left = 0) #Sets lower x-limit to zero
        self.candAx.set_title(path)
        self.candAx.legend()

    def emptycanvas(self):
    
        fig = Figure(figsize = (8,6))
        ax = fig.add_subplot(111)
        ax.set_xlabel("Empty")
        ax.set_ylabel("Empty")
        ax.set_xlim(left = 0) #Sets lower x-limit to zero
        self.canvas = FigureCanvasTkAgg(fig, self)
        self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)