import io
import re

import chess
import torch
from mine.get_squares import obtine_patrate

import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nume_model = "jonglet/deit_tiny"
# nume_model = "jonglet/mobile_vit"
# nume_model = "jonglet/microsoft_cvt"
# nume_model = "jonglet/clasif"

model = AutoModelForImageClassification.from_pretrained(nume_model)
model.eval()
model.to(device)

processor = AutoImageProcessor.from_pretrained(nume_model)


def determina_fen_imagine(img, tura: bool = chess.WHITE, colturi=None):
    if colturi is None:
        patrate = obtine_patrate(img, tura)
    elif isinstance(colturi, np.ndarray):
        patrate = obtine_patrate(img, tura, colturi)
    else:
        raise TypeError(
            f"Tipul {type(colturi)} nu este suportat. Tipurile suportate sunt:"
            "None, np.ndarray"
        )
    return get_fen(patrate)


@torch.no_grad()
def get_fen(patrate):
    fen_output = io.StringIO()
    inputs = [processor(patrat, return_tensors="pt").to(device) for patrat in patrate]
    for i in range(8):
        for j in range(8):
            input = inputs[i * 8 + j]
            outputs = model(**input).logits
            predicted = outputs.argmax(-1).item()
            fen_output.write(model.config.id2label[predicted])
        fen_output.write("/")

    return extended_2_fen(fen_output.getvalue()[0:-1])


def extended_2_fen(ext_fen):
    """
    Transforma 1-urile consecutive in cifre.
    Ex: p11111np/Q1qP11R1 -> p5np/Q1qP2R1
    :param ext_fen: fen-ul care trebuie schimbat
    :return: fen-ul standard
    """
    fen = io.StringIO()
    expresie = re.compile(r"1{2,}")
    for rank in ext_fen.split("/"):
        rank = expresie.sub(lambda x: str(len(x.group(0))), rank)
        fen.write(rank)
        fen.write("/")

    return fen.getvalue()[0:-1]


def get_move(fen):
    """
    Genereaza mutarea pentru a ajunge la fenul dat
    :param fen:
    :return: mutarea in format uci
    """
    board = chess.Board(fen)
    for mutare in board.legal_moves:
        board.push(mutare)
        if board.board_fen() == fen:
            return mutare.uci()
        else:
            board.pop()
    raise ValueError(f"Nu s-a putut genera mutarea pentru fen-ul {fen}")
