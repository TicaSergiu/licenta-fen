from PIL import Image

import chess
import cv2

import numpy as np
from matplotlib import pyplot as plt

from chesscog import find_corners
from chesscog.warp_img import crop_square, warp_chessboard_image

from mine.pozitii_tabla import Patrat


def fen_2_extended(fen_to_replace: str) -> str:
    """
    Modifica fen-ul eliminand '/' si inlocuim cifrele cu 1 de cate ori este cifra
    :param fen_to_replace:
    :return: fen-ul modificat
    """
    for numar in range(2, 9):
        fen_to_replace = fen_to_replace.replace(str(numar), "1" * numar)
    return fen_to_replace.replace("/", "")


def obtine_patrate(imagine, tura: bool, colturi=None) -> list:
    if isinstance(imagine, str):
        try:
            imagine = cv2.imread(imagine)
        except Exception as e:
            raise ValueError(
                f"Nu s-a gasit imaginea la locatia {imagine}") from e

    elif isinstance(imagine, Image.Image):
        imagine = np.array(imagine)

    elif isinstance(imagine, np.ndarray):
        pass
    else:
        raise TypeError(
            f"Tipul {type(imagine)} nu este suportat. Tipurile suportate sunt:"
            "str, np.ndarray, PIL.Image.Image")
    if colturi is None:
        return [square for square in _obtine_patrate(imagine, tura)]
    else:
        return [square for square in _obtine_patrate_colturi(imagine, tura, colturi)]


def _obtine_patrate_colturi(imagine: np.ndarray, turn: bool, colturi: np.ndarray):
    imagine = warp_chessboard_image(imagine, colturi)
    pozitii = Patrat().get_next()
    for _ in range(8):
        for _ in range(8):
            square = crop_square(
                imagine, chess.parse_square(next(pozitii)), turn
            )
            yield square


def _obtine_patrate(imagine: np.ndarray, turn: bool):
    """
    Obtine patratele dintr-o imagine
    :param imagine: imaginea din care se obtin patratele
    :param turn: True daca imaginea este din partea pieselor albe, False altfel
    :return: lista de patrate
    """
    corners = find_corners(imagine)
    imagine = warp_chessboard_image(imagine, corners)

    pozitii = Patrat().get_next()
    for _ in range(8):
        for _ in range(8):
            square = crop_square(
                imagine, chess.parse_square(next(pozitii)), turn
            )
            yield square


def show_patrate(patrate):
    plt.figure(figsize=(12, 12))
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(cv2.cvtColor(patrate[i], cv2.COLOR_BGR2RGB))
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    tura = chess.WHITE
    patrate = obtine_patrate("../date_test/i1.jpeg", tura)

    show_patrate(patrate)

# def main(index: int):
#     start = time.perf_counter()
#     imagini = glob.glob("*.png")
#     imagini.sort()
#
#     fens_and_turn = glob.glob("*.json")
#     fens_and_turn.sort()
#
#     fen = json.load(open(fens_and_turn[index]))["fen"]
#     turn = json.load(open(fens_and_turn[index]))["white_turn"]
#     fen = modifica_fen(fen)
#     locatie = str(imagini[index])
#     img = cv2.imread(locatie)
#
#     corners = find_corners(img)
#     img = chesscog.warp_img.warp_chessboard_image(img, corners)
#
#     pozitii = Patrat().get_next()
#     for rank in range(8):
#         for col in range(8):
#             square = chesscog.warp_img.crop_square(
#                 img, chess.parse_square(next(pozitii)), turn
#             )
#             piesa = fen[rank * 8 + col]
#             nume_fis = uuid.uuid4().hex
#             # print(f"Piesa: {piesa}, locatie: data/{piesa}/{nume_fis}.png")
#             cv2.imwrite(f"../../data/{piesa}/{nume_fis}.png", square)
#
#     return time.perf_counter() - start, locatie
