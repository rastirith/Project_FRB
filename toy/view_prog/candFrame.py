
import tkinter as tk
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.patches as patches
import time
#from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#from globalDef import candPlot
np.set_printoptions(linewidth = 100)
fontx = ('Helvetica', 8)
class_colours = ["b", "m", "r"]

class candClass(tk.Frame):
    
    review_ind_frb = 0      # Keeps track of which index in the FRB-folder the user is currently at
    review_ind_nfrb = 0     # Keeps track of which index in the nFRB-folder the user is currently at
    choice = 0              # Variable indicating whether user is displaying candidates or non-candidates
    current_choice = []     # Folder containing paths of the current FRB or nFRB files.
    xref = 0                # User's chosen x-dimension
    yref = 0                # User's chosen y-dimension
    frb_paths = []          # Paths of all FRB candidates
    nfrb_paths = []         # Paths of all non-candidates
    excellent = []          # All points with an "excellent" classification in the current file
    exc_paths = []          # Paths to all files containing candidates with an "excellent" classification
    good = []               # All points with a "good" classification in the current file
    good_paths = []         # Paths to all files containing candidates with an "good" classification
    least_acc = []          # All points with a "least acceptable" classification in the current file
    least_paths = []        # Paths to all files containing candidates with an "least acceptable" classification
    candidates = []         # Array of arrays, where each position index corresponds to a separate candidate.
    xmax = 0                # Upper x-limit for zoomed S/N plots
    ymax = 0                # Upper y-limit for zoomed S/N plots
    
    axes = []               # Folder containing the axes of the displayed figure, needed to that a single axis can be hidden/removed
    grayAx = None           # Axis displaying all the gray points
    candAx = None           # Axis of underlying figure   
    candFig = None          # Underlying figure
    showVar = 0             # Variable indicating whether classification are being highlighted or not. 1 = Shown, 0 = Hidden.
    markerVar = 0           # Variable indicating whether a marker around a candidate is being shown
    currMarkedInd = None    # Variable containing the index in the candidates array of the currently marked candidate
    currMarkedLims = []     # Array containing the limit of the currently marked candidate
    rect = None             # Rectangle patch shown when clicking on a highlighted cluster

    
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

        menuBar = tk.Menu(self.controller)      # Main horizontal menu bar
        fileMenu = tk.Menu(self.controller)     # First drop down menu
        viewMenu = tk.Menu(self.controller)     # Second drop down menu
        classMenu = tk.Menu(self.controller)    # Sub menu to viewMenu, contains classifications to display
        
        menuBar.add_cascade(label = 'File', menu = fileMenu)
        menuBar.add_cascade(label = 'View', menu = viewMenu)
        viewMenu.add_cascade(label = 'Classifications shown', menu = classMenu)

        classMenu.add_command(label = 'Excellent', command = lambda: self.showClasses(0))
        classMenu.add_command(label = 'Good', command = lambda: self.showClasses(1))
        classMenu.add_command(label = 'Least acceptable', command = lambda: self.showClasses(2))
        classMenu.add_command(label = 'All', command = lambda: self.showClasses(3))
        
        self.controller.config(menu=menuBar)
        
    # Method defining the Labels (including the canvas) part of the GUI
    def labels(self):
        self.emptycanvas()
        tk.Label(self, text = "Y:", font = "Helvetica 9 bold").grid(row = 11, column = 54)  # Y-label for Y axis radiobuttons
        tk.Label(self, text = "X:", font = "Helvetica 9 bold").grid(row = 12, column = 54)  # X-label for X axis radiobuttons
        
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
                self.xref = x.get()
                self.yref = y.get()
        
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
        self.show_button = tk.Button(self, text = "Show", command = lambda: self.canvasPlot())
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
        self.class_btn = tk.Button(self, text = "Show classifications", command = lambda: self.candData())
        self.class_btn.config(height = 2, width = 20)
        self.class_btn.grid(row = 18, column = 51, columnspan = 6, rowspan = 4, pady = (5,5), padx = (15,0))
        
        # Button to zoom in on a marked candidate
        self.zoomIn_btn= tk.Button(self, text = "Zoom in", command = lambda: self.zoom(1))
        self.zoomIn_btn.grid(row = 22, column = 51, rowspan = 4, pady = (5,5), padx = (15,0))
        self.zoomIn_btn.config(height = 2, width = 9)
        self.zoomOut_btn= tk.Button(self, text = "Zoom out", command = lambda: self.zoom(0))
        self.zoomOut_btn.grid(row = 26, column = 51, rowspan = 4, pady = (5,5), padx = (15,0))
        self.zoomOut_btn.config(height = 2, width = 9, relief = 'sunken')
      
    # Method do show candidates of specific class chosen by user in the drop down menu
    def showClasses(self, var):
        if var == 0:        # Shows candidates of 'excellent' classification
            del self.current_choice
            self.current_choice = self.exc_paths    # Updates the path folder with all paths to 'excellent' candidates
            self.choice = 2                         # Exception setting for canvasPlot
            self.review_ind_frb = 0                 # Starts index over
            self.newLabArr()                        # Runs the initiation method for this new path space
            self.canvasPlot()                       # Plots the results
        elif var == 1:      # Shows candidates of 'good' classification
            del self.current_choice
            self.current_choice = self.good_paths
            self.choice = 2
            self.review_ind_frb = 0
            self.newLabArr()
            self.canvasPlot()
        elif var == 2:      # Shows candidates of 'least acceptable' classification
            del self.current_choice
            self.current_choice = self.least_paths
            self.choice = 2
            self.review_ind_frb = 0
            self.newLabArr()
            self.canvasPlot()
        else:               # Shows all candidates
            del self.current_choice
            self.current_choice = self.frb_paths
            self.choice = 1
            self.review_ind_frb = 0
            self.newLabArr()
            self.canvasPlot()
     
    # Method to zoom into a marked candidates
    def zoom(self, var):
        if var == 1:    # Changes x and y limits to zoom in on the marked candidate
            self.zoomIn_btn.config(relief="sunken")
            self.zoomOut_btn.config(relief="raised")
            self.candAx.set_xlim(self.currMarkedLims[0], self.currMarkedLims[1])
            self.candAx.set_ylim(self.currMarkedLims[2], self.currMarkedLims[3])
        else:           # Zooms out from a marked condidates by returning x and y limits to normal
            self.zoomOut_btn.config(relief="sunken")
            self.zoomIn_btn.config(relief="raised")
            if (self.xref == 2 and self.yref == 2): # If one of the axis contains SN data then start this axis at SN = 8
                self.candAx.set_xlim(8, self.xmax)
                self.candAx.set_ylim(8, self.ymax)
            elif (self.xref == 2):
                self.candAx.set_xlim(8, self.xmax)
                self.candAx.set_ylim(0, self.ymax)
            elif (self.yref == 2):
                self.candAx.set_xlim(0, self.xmax)
                self.candAx.set_ylim(8, self.ymax)
            else:
                self.candAx.set_xlim(0, self.xmax)
                self.candAx.set_ylim(0, self.ymax)
            
        self.canvas = FigureCanvasTkAgg(self.candFig, self) # Updates the canvas
        self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 500)
        self.canvas.mpl_connect("button_press_event", self.callback)   # Connects canvas to mouse click events  
        
    # Method to display the next image in the path space
    def Right(self):
        # Assigns the approriate array index
        if self.choice == 1 or self.choice == 2:    
            review_ind = self.review_ind_frb
        else:
            review_ind = self.review_ind_nfrb
        
        if (review_ind >= (len(self.current_choice) - 1)):  # If at the end of the array don't try to go further
            print("Reached end of files.")
        else:
            if self.choice == 1 or self.choice == 2:    # Updates the approriate path index
                self.review_ind_frb += 1
            else:
                self.review_ind_nfrb += 1
            self.newLabArr()        # Initiation commands
            if self.showVar == 1:   # Goes in here if the next plot displayed should contain candidate highlights
                self.showVar = 0
                self.drawGray()
                self.candData()
                self.canvas = FigureCanvasTkAgg(self.candFig, self)
                self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
                self.canvas.mpl_connect("button_press_event", self.callback)
            else:                   # If no candidate highlights then goes in here
                self.drawGray()
                self.canvas = FigureCanvasTkAgg(self.candFig, self)
                self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)

    # Same method as above, but switches to the previous plot instead
    def Left(self):
        if self.choice == 1 or self.choice == 2:
            review_ind = self.review_ind_frb
        else:
            review_ind = self.review_ind_nfrb
        
        if (review_ind <= 0):   # If at the beginning on the array don't try to go further back
            print("At the first file already.")
        else:
            if self.choice == 1 or self.choice == 2:
                self.review_ind_frb -= 1
            else:
                self.review_ind_nfrb -= 1
            self.newLabArr()
            if self.showVar == 1:
                self.showVar = 0
                self.drawGray()
                self.candData()
                self.canvas = FigureCanvasTkAgg(self.candFig, self)
                self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
                self.canvas.mpl_connect("button_press_event", self.callback)
            else: 
                self.drawGray()
                self.canvas = FigureCanvasTkAgg(self.candFig, self)
                self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
            
    #Exits client
    def client_exit(self):
        exit()   
      
    #Method to put image in appropriate folder, make a note where that is, and change the image on display
    def display_choice(self, xchoice):
        self.choice = xchoice
        
        if (xchoice == 1):      # Goes in here to display FBR candidates
            self.frb_button.config(relief="sunken")
            self.nfrb_button.config(relief="raised")
            self.current_choice = self.frb_paths
            self.newLabArr()
            self.classArrays()
            self.canvasPlot()
        else:                   # Goes in here to display non-FRB candidates
            self.nfrb_button.config(relief="sunken")
            self.frb_button.config(relief="raised")
            self.current_choice = self.nfrb_paths
            self.canvasPlot()
    
    # Method to initialise a plot on the canvas
    def canvasPlot(self):
        if self.choice == 1:    # Sets the path space to the one containing FRB paths
            self.current_choice = self.frb_paths
        elif self.choice == 2:  # Leaves the path space unchanged, exception case
            pass
        else:                   # Sets the path space to the one containing non-FRB paths
            self.current_choice = self.nfrb_paths
        
        if self.showVar == 1:   # Goes in here if the plot to be displayed should include candidate highlights
            self.showVar = 0
            self.drawGray()
            self.candData()
            if self.markerVar == 1: # Goes in here if a candidate is marked and should remain marked on the new axis
                self.callback(1)
            else:           # Goes in here if a highlighted candidate is not marked     
                self.canvas = FigureCanvasTkAgg(self.candFig, self)
                self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
                self.canvas.mpl_connect("button_press_event", self.callback)
        else:   # Goes in here if the plot to be displayed should not include candidate highlights
            self.drawGray()
            self.canvas = FigureCanvasTkAgg(self.candFig, self)
            self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)

    # Method to either display or remove candidate highlights
    def candData(self):
        
        if self.showVar == 0:   # Goes in here if classification button is pressed to display candidates
            
            self.class_btn.config(text = "Hide classifications")    # Updates the button text
            self.grayAx.set_label('RFI/Background')                 # Updates the legend labels
    
            if len(self.excellent) > 0:     # If 'excellent' candidates exists, initialise the corresponding axis here
                self.axes.append(self.candAx.scatter(self.excellent[:,self.xref], self.excellent[:,self.yref], color = "r", alpha = 1, vmin = -1, s = 6, label = "Excellent"))
            if len(self.good) > 0:          # If 'good' candidates exists, initiate the corresponding axis here
                self.axes.append(self.candAx.scatter(self.good[:,self.xref], self.good[:,self.yref], color = "m", alpha = 1, vmin = -1, s = 6, label = "Good"))
            if len(self.least_acc) > 0:     # If 'least acceptable' candidates exists, initiate the corresponding axis here
                self.axes.append(self.candAx.scatter(self.least_acc[:,self.xref], self.least_acc[:,self.yref], color = "b", alpha = 1, vmin = -1, s = 6, label = "Least acceptable"))
            self.candAx.legend()            # Updates legend
            self.canvas = FigureCanvasTkAgg(self.candFig, self)     # Updates the canvas
            self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
            self.canvas.mpl_connect("button_press_event", self.callback)    # Binds mouse click event

            self.showVar = 1    # Updates variable
        else:       # Goes in here if classification button is pressed to hide candidates
            if len(self.axes) > 0:
                self.class_btn.config(text = "Show classifications")
                for i in self.axes:     # Removes candidate axes
                    i.remove()
                self.grayAx.set_label('Events') # Updates legend label
                self.candAx.legend()            # Updates legend
            else:
                print("No candidates highlighted.")

            self.canvas = FigureCanvasTkAgg(self.candFig, self)     # Updates canvas
            self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
            del self.axes
            self.axes = []      # Sets the axes array to be an empty array
            self.showVar = 0    # Updates variable
        
    # Method to import data from .dat file and draw all points without highlighted candidates  
    def drawGray(self):
        
        if self.choice == 1 or self.choice == 2:    # Chooses the index correspoding to the FRB path space
            review_ind = self.review_ind_frb
        else:                                       # Chooses the index correspoding to the non-FRB path space
            review_ind = self.review_ind_nfrb
        
        path = self.current_choice[review_ind]      # Path to .dat file to be plotted
        
        #Imports data from the .dat file located at the specified path
        file = open(path,'r')
        data = np.fromfile(file,np.float32,-1) 
        c = data.reshape((-1,4))
        file.close()
    
        #Defines labels for the axes corresponding to the four different sets of data available
        axislabels = ["DM", "Time", "s/n", "Width"]
        columns = np.hsplit(c,4)    # x- and yref values dm=0, time=1, s/n=2, width=3
        self.xmax = np.amax(columns[self.xref])
        self.ymax = np.amax(columns[self.yref])
        
        #Creates the figure to be displayed. xref and yref corresponding to the chosen x and y values to be displayed
        self.candFig = Figure(figsize = (8,6))
        self.candAx = self.candFig.add_subplot(111)
        self.candAx.set_xlabel(axislabels[self.xref])
        self.candAx.set_ylabel(axislabels[self.yref])
        self.grayAx = self.candAx.scatter(columns[self.xref], columns[self.yref], color = "0.7", alpha = 1, vmin = -1, s = 6, label = "Events")
        self.candAx.set_xlim(left = 0) # Sets lower x-limit to zero
        self.candAx.set_title(path.split('\\')[-1]) # Title is the file name
        self.candAx.legend()


    # Method to draw on empty canvas before any plot has been chosen
    def emptycanvas(self):
    
        fig = Figure(figsize = (8,6))
        ax = fig.add_subplot(111)
        ax.set_xlabel("Empty")
        ax.set_ylabel("Empty")
        ax.set_xlim(left = 0) # Sets lower x-limit to zero
        self.canvas = FigureCanvasTkAgg(fig, self)  # Updates canvas
        self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
        
    # Method that conducts some operations that need to be done every time a new image is to be displayed
    def newLabArr(self):
        
        candFile = self.current_choice[self.review_ind_frb].replace('.dat','_c.csv')    # Path to the candidates data files
        dataset = pd.read_csv(candFile)     # Reads the candidate data file
    
        X = dataset.iloc[:,0:6].values
        X = np.array(X)

        self.excellent = []     # Array containing all 'excellent' data points
        self.good = []          # Array containing all 'good' data points
        self.least_acc = []     # Array containing all 'least acceptable' data points
        
        for i in range(len(X[:,4])):
            if X[:,4][i] == 3:  # If the point has an 'excellent' label, add it to the corresponding array
                self.excellent.append(X[i])
            elif X[:,4][i] == 2:    # If the point has a 'good' label, add it to the corresponding array
                self.good.append(X[i])
            elif X[:,4][i] == 1:    # If the point has a 'least acceptable' label, add it to the corresponding array
                self.least_acc.append(X[i])
        self.excellent = np.array(self.excellent)   # Set arrays to Numpy arrays
        self.good = np.array(self.good)
        self.least_acc = np.array(self.least_acc)
        
        numCand = len(set(X[:,5]))  # Counts the number of separate candidate cluster in the file
        self.candidates = []        # Array of arrays where each index contains a candidate cluster
        
        for i in range(numCand):        # For loop that separates the different clusters and puts them in the candidates array
            temp = []
            for k in range(len(X[:,5])):    # Goes through all individual points in the _c.csv file
                if X[:,5][k] == i:
                    temp.append(X[k])
            temp = np.array(temp)
            self.candidates.append(temp)    # Puts array of individual points of certain cluster into array
            
        self.candidates = np.array(self.candidates)
        
        if self.markerVar == 1:             # If there was a marker then remove it as new file is being displayed
            self.rect.remove()
            self.currMarkedInd = None
            self.markerVar = 0              # Resets marker indicator 
           
    # Method to go through all FRB files to check whether the files contains candidates of certain classification
    def classArrays(self):
        for i in range(len(self.frb_paths)):    # Loops through all paths in the FRB folder 
            exc_var = False                     # False = Does not contain excellent cand
            good_var = False                    # True = Does contain such a cand.
            least_var = False
            
            candFile = self.frb_paths[i].replace('.dat','_c.csv')
            dataset = pd.read_csv(candFile)
        
            X = dataset.iloc[:,0:6].values  # Collects the candidate data
            X = np.array(X)
            
            for i in range(len(X[:,4])):    # Goes through the class labels of each point
                if (X[:,4][i] == 3 and exc_var == False):       # Goes in here the first time a point has an 'excellent' label
                    self.exc_paths.append(candFile.replace('_c.csv','.dat'))
                    exc_var = True
                elif (X[:,4][i] == 2 and good_var == False):    # Goes in here the first time a point has a 'good' label
                    self.good_paths.append(candFile.replace('_c.csv','.dat'))
                    good_var = True
                elif (X[:,4][i] == 1 and least_var == False):   # Goes in here the first time a point has a 'least acceptable' label
                    self.least_paths.append(candFile.replace('_c.csv','.dat'))
                    least_var = True
        
    # Callback function for mouse click events. 
    def callback(self, event):

        # Goes in here in there is no marker currently being shown        
        if self.markerVar == 0:
            for i in range(len(self.candidates)):       # Loops through all the candidates
                x = self.candidates[i][:,self.xref]     # xy fig. coords.   
                y = self.candidates[i][:,self.yref]
                
                xy_pixels = self.candAx.transData.transform(np.vstack([x,y]).T) # Transforms xy fig. coords. to pixels
                xpix, ypix = xy_pixels.T
                
                maxX = np.amax(xpix)    # Highest pixel value of the candidate in the x-direction
                minX = np.amin(xpix)    # Lowest pixel value of the candidate in the x-direction
                maxY = np.amax(ypix)    # Highest pixel value of the candidate in the y-direction
                minY = np.amin(ypix)    # Highest pixel value of the candidate in the y-direction
                
                # Goes in here if the mouse click is within a rectangle of dimensions of the cluster with a 5 pixel padding
                if ((event.y < maxY + 5) and (event.y > minY - 5) and (event.x < maxX + 5) and (event.x > minX - 5)):
                    
                    botX_p = minX - 5   # Lowest accepted pixel value in the x-direciton for mouse clicks
                    botY_p = minY - 5   # Lowest accepted pixel value in the y-direciton for mouse clicks
                    topX_p = maxX + 5   # Highest accepted pixel value in the x-direciton for mouse clicks
                    topY_p = maxY + 5   # Highest accepted pixel value in the y-direciton for mouse clicks
    
                    # Transforms pixel values to fig. coords. Needed for patches.Rectangle positions
                    botX_co, botY_co = self.candAx.transData.inverted().transform(np.vstack([botX_p,botY_p]).T).T
                    topX_co, topY_co = self.candAx.transData.inverted().transform(np.vstack([topX_p,topY_p]).T).T
                    
                    # Adds the figure coordinate limits for the candidate to the currMarkedLims array for plotting
                    self.currMarkedLims = []
                    self.currMarkedLims.append(botX_co)
                    self.currMarkedLims.append(topX_co)
                    self.currMarkedLims.append(botY_co)
                    self.currMarkedLims.append(topY_co)
                    
                    # Draws a rectangle around the candidate
                    self.rect = patches.Rectangle((botX_co,botY_co),topX_co - botX_co,topY_co - botY_co,linewidth=2,edgecolor='k',facecolor='none')
                    
                    # Add the patch to the Axes
                    self.candAx.add_patch(self.rect)
                    self.canvas = FigureCanvasTkAgg(self.candFig, self)
                    self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
                    self.canvas.mpl_connect("button_press_event", self.callback)
                    self.markerVar = 1      # Indicates that a marker is currently being shown
                    self.currMarkedInd = i     # Indicates which cluster is being marked
                    break
                
        # Goes in here if a marker is currently being shown
        else:
            self.rect.remove()  # Removes the old marker
            if event == 1:  # Special case, used when the axes are changed whilst displaying a rectangle
                x = self.candidates[self.currMarkedInd][:,self.xref]       # xy fig. coords.   
                y = self.candidates[self.currMarkedInd][:,self.yref]
                
                xy_pixels = self.candAx.transData.transform(np.vstack([x,y]).T) # Transforms xy fig. coords. to pixels
                xpix, ypix = xy_pixels.T
                
                maxX = np.amax(xpix)    # Commented on the similar case above
                minX = np.amin(xpix)
                maxY = np.amax(ypix)
                minY = np.amin(ypix)              
                botX_p = minX - 5
                botY_p = minY - 5
                topX_p = maxX + 5
                topY_p = maxY + 5

                # Transforms pixel values to fig. coords. Needed for patches.Rectangle positions
                botX_co, botY_co = self.candAx.transData.inverted().transform(np.vstack([botX_p,botY_p]).T).T
                topX_co, topY_co = self.candAx.transData.inverted().transform(np.vstack([topX_p,topY_p]).T).T

                self.currMarkedLims = []                # Commented on the similar case above
                self.currMarkedLims.append(botX_co)
                self.currMarkedLims.append(topX_co)
                self.currMarkedLims.append(botY_co)
                self.currMarkedLims.append(topY_co)
                    
                self.rect = patches.Rectangle((botX_co,botY_co),topX_co - botX_co,topY_co - botY_co,linewidth=2,edgecolor='k',facecolor='none')
                
                # Add the patch to the Axes
                self.candAx.add_patch(self.rect)
                self.canvas = FigureCanvasTkAgg(self.candFig, self)
                self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
                self.canvas.mpl_connect("button_press_event", self.callback)
            
            elif (len(self.candidates) > 1):    # Goes in here if there are multiple candidates in the file
                for i in range(len(self.candidates)):       # Loops through the candidates
                    x = self.candidates[i][:,self.xref]
                    y = self.candidates[i][:,self.yref]
            
                    xy_pixels = self.candAx.transData.transform(np.vstack([x,y]).T)
                    xpix, ypix = xy_pixels.T
                    
                    maxX = np.amax(xpix)
                    minX = np.amin(xpix)
                    maxY = np.amax(ypix)
                    minY = np.amin(ypix)
                    
                    # Goes in here mouse click is within range of the candidate and it is not the currently highlighted one
                    if ((event.y < maxY + 5) and (event.y > minY - 5) and (event.x < maxX + 5) and (event.x > minX - 5) and (self.currMarkedInd != i)):
                        botX_p = minX - 5
                        botY_p = minY - 5
                        topX_p = maxX + 5
                        topY_p = maxY + 5
        
                        botX_co, botY_co = self.candAx.transData.inverted().transform(np.vstack([botX_p,botY_p]).T).T
                        topX_co, topY_co = self.candAx.transData.inverted().transform(np.vstack([topX_p,topY_p]).T).T
        
                        self.currMarkedLims = []
                        self.currMarkedLims.append(botX_co)
                        self.currMarkedLims.append(topX_co)
                        self.currMarkedLims.append(botY_co)
                        self.currMarkedLims.append(topY_co)
                        
                        self.rect = patches.Rectangle((botX_co,botY_co),topX_co - botX_co,topY_co - botY_co,linewidth=2,edgecolor='k',facecolor='none')
                        
                        # Add the patch to the Axes
                        self.candAx.add_patch(self.rect)
                        self.markerVar = 1
                        self.currMarkedInd = i
                        break
                    else:   # Goes in here if the mouse click is outside the candidates or it was on an already marked one
                        self.currMarkedInd = None
                        self.markerVar = 0
            else:   # Goes in here if the file doesn't contain multiple candidates
                self.currMarkedInd = None
                self.markerVar = 0
                
            self.canvas = FigureCanvasTkAgg(self.candFig, self) # Updates the canvas
            self.canvas.get_tk_widget().grid(row = 0, column = 0, columnspan = 50, sticky = "nw", rowspan = 50)
            self.canvas.mpl_connect("button_press_event", self.callback)
