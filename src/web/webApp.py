from flask import Flask, make_response, render_template, request, redirect
import os

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    print(request.method)
    if os.path.exists("web/static/temp.jpg"):
        os.remove("web/static/temp.jpg")
        return redirect(request.url)
    match request.method:
        case "POST":
            f = request.files["file"]
            if f.filename == "":
                return redirect(request.url)
            f.filename = "temp.jpg"
            f.save(f"web/static/{f.filename}")
            resp = make_response(render_template("index.html"))
            resp.set_cookie("img", "true", 300)
            return resp
        # return schimba_imaginea()
        case "GET":
            return render_template("index.html")


def schimba_imaginea():
    f = request.files["file"]
    if f.filename == "":
        return redirect(request.url)
    f.filename = "temp.jpg"
    f.save(f"web/static/{f.filename}")
    return render_template("index.html")
