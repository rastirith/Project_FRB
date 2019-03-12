import glob
import os
import tkinter as tk

from mainFrame import main_frame
from previewFrame import preview_frame
from candFrame import candClass

#*****MAIN CLASS*****#
class start_app(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        # The container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.grid()

        self.frames = {}
        for F in (main_frame, candClass, preview_frame):    # Creating all frames to be layered
            page_name = F.__name__
            frame = F(master=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("main_frame")       # Shows the 'main_frame' first

    # Method that lifts the 'page_name' fram to the top, i.e. displays it
    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()
        frame.focus_set()
        
        menubar = frame.menuobj()
        self.configure(menu=menubar)
       
# MAIN STARTS HERE
def main():
    
    # Input directories as defined in relation to the work. dir.
    idir_path = os.getcwd() + "\\idir"
    odir_path = os.getcwd() + "\\odir"
    main_frame.odir_path = odir_path
    
    try:    #Only creates folders if they don't already exist
        os.mkdir(idir_path)
        os.mkdir(odir_path)
    except:
        pass
        
    completeName = os.path.join(odir_path,"results.txt")
    main_frame.result_file=open(completeName,"a")

    source_paths = []       #List of filenames to be viewed in program
    
    # Puts paths of all unclassified .dat files in 'source_paths' array
    for file in glob.glob(idir_path + "/" + "*.dat"):
        source_paths.append(file)
    main_frame.source_paths = source_paths      # Merges the source_paths class objects with source_paths
    preview_frame.source_paths = source_paths
    candClass.source_paths = source_paths
    
    for file in glob.glob(idir_path + "\\candidates\\" + "*.dat"):  # Saving all paths to .dat files in FRB folder in candClass
        candClass.frb_paths.append(file)

    for file in glob.glob(odir_path + "\\no_frb\\" + "*.dat"):      # Saving all paths to .dat files in non-FRB folder in candClass
        candClass.nfrb_paths.append(file)
    
    main_frame.length = len(source_paths)

    app = start_app()
    app.mainloop()
   
    main_frame.result_file.close()
   
if __name__ == '__main__':
    main()