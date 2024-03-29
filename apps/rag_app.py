from tkinter import *
from tkinter import filedialog

from src.Rag.AuroraRag import AuroraRag


class App(Tk):
    def __init__(self):
        super().__init__()
        self.geometry("700x550")
        self.title("Rag App")
        self.text_box()
        self.add_load_pdf_button()
        self.my_file = ""
        self.rag = None
        self.add_search_panel()

    def add_search_panel(self):
        self.search_entry = Entry(self)
        self.search_entry.insert(0, "Search query")
        self.search_entry.config(state=DISABLED)
        self.search_entry.bind("<Button-1>", self.click)
        self.search_entry.bind("<Return>", lambda event: self.search())
        self.search_button = Button(self, text="Search", command=self.search)
        self.search_entry.pack(side=LEFT, padx=5, expand=True, fill=X)
        self.search_button.pack(side=LEFT, padx=5)

    def add_load_pdf_button(self):
        self.load_pdf_button = Button(self, text="Get PDF", command=self.get_pdf)
        self.load_pdf_button.pack(pady=20)

    def text_box(self):
        self.rag_text = Text(self, width=80, height=20)
        self.rag_text.pack(pady=20)

    def get_pdf(self):
        self.my_file = filedialog.askopenfilename(initialdir = "", title = "Select PDF", filetypes = (("PDF files", "*.pdf"), ("all files", "*.*")))
        if self.my_file:
            self.rag_text.insert(END, self.my_file)

    def search(self):
        self.rag_text.insert(END, "You are looking for: " + self.search_entry.get())
        if self.my_file == "":
            self.rag_text.insert(END, "No file found, make sure that the file exists and readable.")
        else:
            if self.search_entry.get() == "":
                self.rag_text.insert(END, "Please enter a search query")
                return  
            self.rag = AuroraRag(self.my_file, "all-mpnet-base-v2")
            self.rag_text.insert(END, "creating rag pipeline to search your file...")
            
            results = self.rag.search(self.search_entry.get())
            self.rag_text.delete("1.0", END)
            text = self.rag.display_results(results)

            self.rag_text.insert(END, text)

    def click(self, event):
        self.search_entry.config(state=NORMAL)
        self.search_entry.delete(0, END)


if __name__ == "__main__":
    app = App()
    app.mainloop()
        
        