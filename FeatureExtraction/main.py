
from  View import View
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    View(root).grid()#side="top", fill="both", expand=True)
    root.title("Feature Extraction")
    #root.geometry("650x200")
    root.mainloop()

