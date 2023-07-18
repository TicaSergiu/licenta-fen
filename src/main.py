import sys
import cv2
import numpy as np
import argparse
import pathlib
import os
from time import perf_counter
from PIL import Image
import webbrowser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert image to FEN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("src", help="Locatia imaginii", type=pathlib.Path, nargs="?")

    parser.add_argument("--gui", help="Porneste interfata grafica", action="store_true")

    parser.add_argument(
        "-q", "--quiet", help="Afiseaza doar rezultatul", action="store_true"
    )

    parser.add_argument(
        "-t", "--tura", help="Culoarea jucatorului(a sau n)", default="a"
    )

    parser.add_argument(
        "-b",
        "--browser",
        action="store_true",
        help="Deschide in lichess sau fisierul la sfarsitul executiei",
    )

    parser.add_argument(
        "-l",
        "--list",
        nargs="*",
        action="append",
        help="Coordonatele celor 4 colturi(ordinea nu conteaza), doar cand este specificata o singura imagine\
                                De ex: -l 0,0 120,683 423,23 423,683",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Numele fisierului cu rezultate cand este folosit -d",
        type=pathlib.Path,
        default="output.txt",
    )

    args = parser.parse_args()

    if args.gui:
        from gui.fereastra import Fereastra

        video = cv2.VideoCapture(0)
        fereastra = Fereastra(video)
        fereastra.mainloop()
        sys.exit()

    argument: pathlib.Path = args.src
    if argument is None:
        parser.print_help()
        sys.exit()

    from transformer.get_fen import determina_fen_imagine

    if argument.is_dir():
        fens = []
        c = 0
        for fisier in os.scandir(argument):
            try:
                img = Image.open(fisier.path)
                if not args.quiet:
                    print(f"Se proceseaza imaginea {fisier.name} ...")
                c += 1
                fens.append(determina_fen_imagine(img))
            except Exception:
                continue

        if c > 0:
            with open(args.output, "w") as f:
                for fen in fens:
                    f.write(fen + "\n")
            if args.browser:
                webbrowser.open(args.output.name)

        else:
            print("Nu exista imagini in folder")
    elif argument.is_file():
        start = perf_counter()
        img = None
        try:
            img = Image.open(argument)
        except Exception:
            print("Fisierul nu este o imagine")
            sys.exit()

        if not args.quiet:
            print("Se proceseaza imaginea...")

        pct = None
        if args.list:
            argumente = args.list[0]
            if len(argumente) != 4:
                print("Sunt necesare 4 coordonate")
                sys.exit()
            pct = np.zeros((4, 2))
            for i, coordonate in enumerate(argumente):
                x = int(coordonate.split(",")[0])
                y = int(coordonate.split(",")[1])
                if x < 0 or y < 0:
                    print("Coordonatele trebuie sa fie pozitive")
                    sys.exit()
                if x > img.width or y > img.height:
                    print(
                        f"Coordonatele trebuie sa fie mai mici decat dimensiunile imaginii {img.size}"
                    )
                    sys.exit()
                pct[i] = [x, y]

        tura = True if args.tura == "a" else False
        fen = determina_fen_imagine(img, tura, pct)
        stop = perf_counter()
        print(fen)

        if not args.quiet:
            print(f"Procesarea a durat {stop - start:.2f} secunde")
        if args.browser:
            webbrowser.open(f"https://lichess.org/editor/{fen}")
        sys.exit()
    else:
        print("Fisierul nu exista")
        sys.exit()
