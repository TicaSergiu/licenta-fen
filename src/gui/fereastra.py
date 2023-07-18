import cv2
import os
import numpy as np
import webbrowser
import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter
from transformer.get_fen import determina_fen_imagine
from functools import partial


class Fereastra(ctk.CTk):
    AUTO = 1
    MANUAL = 2

    def __init__(self, video):
        super().__init__()

        self.title("Img2Fen")
        self.geometry("200x200")

        self.video = video

        self.imagine_incarcata: Image.Image = Image.new("RGB", (1, 1))
        self.img = None
        self.scale_height = None
        self.scale_witdth = None
        self.filmare = True
        self.camera = False

        self.btn_incarca_imagine = ctk.CTkButton(
            self, text="Incarca imagine", command=self.incarca_imagine
        )
        self.btn_incarca_imagine.place(
            relx=0.5, rely=0.4, anchor=tkinter.CENTER)
        self.btn_foloseste_camera = ctk.CTkButton(
            self, text="Foloseste camera", command=self.arata_camera
        )
        self.btn_foloseste_camera.place(
            relx=0.5, rely=0.6, anchor=tkinter.CENTER)

        self.imagine: ctk.CTkLabel

        self.lbl_tura = ctk.CTkLabel(self, text="Tura jucatorului")

        self.radio_tura = tkinter.IntVar(value=1)
        self.rb_white = ctk.CTkRadioButton(
            self, text="Alb", variable=self.radio_tura, value=1
        )
        self.rb_negru = ctk.CTkRadioButton(
            self, text="Negru", variable=self.radio_tura, value=2
        )

        self.lbl_metoda = ctk.CTkLabel(self, text="Metoda de detectare")

        self.radio_metoda = tkinter.IntVar(value=1)
        self.rb_auto = ctk.CTkRadioButton(
            self,
            text="Auto",
            variable=self.radio_metoda,
            value=1,
            command=self.verifica_metoda,
        )
        self.rb_manual = ctk.CTkRadioButton(
            self,
            text="Manual",
            variable=self.radio_metoda,
            value=2,
            command=self.verifica_metoda,
        )

        loc_img = os.path.dirname(os.path.abspath(__file__))

        self.rotate_left = ctk.CTkImage(
            Image.open(os.path.join(loc_img, "arr_left.png"))
        )
        self.btn_rotate_left = ctk.CTkButton(
            self,
            text="",
            image=self.rotate_left,
            width=30,
            command=partial(self.roteste, 90),
        )

        self.rotate_right = ctk.CTkImage(
            Image.open(os.path.join(loc_img, "arr_right.png"))
        )
        self.btn_rotate_right = ctk.CTkButton(
            self,
            text="",
            image=self.rotate_right,
            width=30,
            command=partial(self.roteste, -90),
        )

        self.btn_get_fen = ctk.CTkButton(
            self, text="FEN", command=self.determina_fen)

        self.lbl_fen = ctk.CTkLabel(self, text="FEN rezultat:")
        self.txt_fen_rezult = ctk.CTkTextbox(self, height=5, width=430)
        self.txt_fen_rezult.insert(
            "0.0", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        self.txt_fen_rezult.configure(state="disabled")

        self.btn_gotolichess = ctk.CTkButton(
            self, text="Analizeaza pe Lichess", command=self.deschide_lichess
        )

        self.lbl_eroare = ctk.CTkLabel(self, text="")

        self.canvas = tkinter.Canvas(self, width=480, height=480)
        self.canvas.bind("<Button-1>", self.pune_punct)
        self.puncte = []

    def arata_camera(self):
        if self.filmare:
            self.camera = True
            _, frame = self.video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            self.imagine_incarcata = frame
            self.img = ImageTk.PhotoImage(frame)

            self.canvas.create_image(0, 0, image=self.img, anchor=tkinter.NW)
            self.canvas.after(20, self.arata_camera)

        self.pune_butoane()

    def pune_punct(self, event):
        if self.radio_metoda.get() == self.AUTO:
            self.radio_metoda.set(2)
        if len(self.puncte) == 4:
            return
        punct = {"x": event.x, "y": event.y}
        self.canvas.create_oval(
            punct["x"] - 6,
            punct["y"] - 6,
            punct["x"] + 2,
            punct["y"] + 2,
            fill="green",
            tags=f"pct-{len(self.puncte) + 1}",
        )
        self.puncte.append(punct)

    def roteste(self, grade):
        if grade == 90:
            self.imagine_incarcata = self.imagine_incarcata.transpose(
                Image.Transpose.ROTATE_90
            )
        else:
            self.imagine_incarcata = self.imagine_incarcata.transpose(
                Image.Transpose.ROTATE_270
            )
        self.scale_witdth = self.imagine_incarcata.width / 480
        self.scale_height = self.imagine_incarcata.height / 480
        self.img = ImageTk.PhotoImage(
            self.imagine_incarcata.resize((480, 480)))

        self.canvas.create_image(0, 0, image=self.img, anchor=tkinter.NW)

    def verifica_metoda(self):
        metoda_aleasa = self.radio_metoda.get()
        if metoda_aleasa == self.AUTO:
            if self.camera:
                print("Filmare true")
                self.filmare = True
                self.arata_camera()
            for i, _ in enumerate(self.puncte):
                self.canvas.delete(f"pct-{i + 1}")
            self.puncte.clear()
        if metoda_aleasa == self.MANUAL:
            self.filmare = False

    def incarca_imagine(self):
        file_types = [("Image files", "*.jpg *.png *.jpeg")]
        filename = ctk.filedialog.askopenfilename(
            title="Select file", filetypes=file_types
        )
        self.imagine_incarcata = Image.open(filename)

        self.scale_height = self.imagine_incarcata.height / 480
        self.scale_witdth = self.imagine_incarcata.width / 480
        self.img = ImageTk.PhotoImage(
            self.imagine_incarcata.resize((480, 480)))

        self.filmare = False
        self.camera = False
        self.pune_butoane()

    def pune_butoane(self):
        self.geometry("730x630")

        self.btn_incarca_imagine.configure(text="Incarca alta imagine")
        self.btn_incarca_imagine.place(relx=0.01, rely=0.01, anchor=tkinter.NW)

        self.imagine = ctk.CTkLabel(self, text="")
        # self.imagine.place(relx=0.95, rely=0.01, anchor=tkinter.NE)

        self.lbl_tura.place(relx=0.01, rely=0.06, anchor=tkinter.NW)
        self.rb_white.place(relx=0.01, rely=0.1, anchor=tkinter.NW)
        self.rb_negru.place(relx=0.01, rely=0.14, anchor=tkinter.NW)

        self.lbl_metoda.place(relx=0.01, rely=0.18, anchor=tkinter.NW)
        self.rb_auto.place(relx=0.01, rely=0.22, anchor=tkinter.NW)
        self.rb_manual.place(relx=0.01, rely=0.26, anchor=tkinter.NW)

        self.btn_get_fen.place(relx=0.01, rely=0.3, anchor=tkinter.NW)

        self.lbl_eroare.place(relx=0.01, rely=0.35, anchor=tkinter.NW)

        self.lbl_fen.place(relx=0.01, rely=0.8, anchor=tkinter.NW)
        self.txt_fen_rezult.place(relx=0.01, rely=0.85, anchor=tkinter.NW)

        self.btn_gotolichess.place(relx=0.65, rely=0.85, anchor=tkinter.NW)

        self.btn_rotate_left.place(relx=0.29, rely=0.78, anchor=tkinter.NW)
        self.btn_rotate_right.place(relx=0.34, rely=0.78, anchor=tkinter.NW)

        self.canvas.place(relx=0.95, rely=0.01, anchor=tkinter.NE)
        self.canvas.create_image(0, 0, image=self.img, anchor=tkinter.NW)

    def determina_fen(self):
        fen = None
        try:
            if self.radio_metoda.get() == self.MANUAL:
                if len(self.puncte) == 4:
                    fen = self.determina_fen_manual()
                else:
                    self.lbl_eroare.configure(
                        text="Trebuie introduse 4 puncte")
                    return
            else:
                fen = self.determina_fen_auto()
            self.txt_fen_rezult.configure(state="normal")
            self.txt_fen_rezult.delete("0.0", "end")
            self.txt_fen_rezult.insert("0.0", fen)
            self.txt_fen_rezult.configure(state="disabled")
            self.lbl_eroare.configure(text="")
        except Exception as e:
            self.lbl_eroare.configure(text="Nu s-au putut determina colțurile")
            return

    def determina_fen_auto(self):
        tura = True if self.radio_tura.get() == 1 else False
        return determina_fen_imagine(self.imagine_incarcata, tura)

    def determina_fen_manual(self):
        lista_puncte = np.zeros((4, 2))
        i = 0
        for punct in self.puncte:
            x = int(punct["x"] * self.scale_witdth)
            y = int(punct["y"] * self.scale_height)
            lista_puncte[i] = [x, y]
            i += 1
        tura = True if self.radio_tura.get() == 1 else False
        return determina_fen_imagine(self.imagine_incarcata, tura, lista_puncte)

    def deschide_lichess(self):
        culoare = "white" if self.radio_tura.get() == 1 else "black"
        webbrowser.open(
            f'https://lichess.org/editor/{self.txt_fen_rezult.get(0.0, "end")}\
            ?color={culoare}'
        )
