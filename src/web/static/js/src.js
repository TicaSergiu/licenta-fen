var points = [];

window.onload = () => {
  let canvas = document.getElementById("canvas");
  canvas.width=350;
  canvas.height=350;
  console.log("da")
}

function esteImagineaIncarcata() {
  let cookie = document.cookie;
  if (cookie.split('=')[1] == 'true') {
    return true;
  }
  return false;
}

canvas.onclick = (event) => {
  if (!esteImagineaIncarcata()) {
    spop({
      template: 'Trebuie incarcata o imagine!',
      style: 'error',
      autoclose: 2500,
      position: 'bottom-center',
      group: 'img_incarcata'
    })
    return;
  }

  const canvas = document.getElementById("canvas");
  const rect = canvas.getBoundingClientRect(),
    scaleX = canvas.clientWidth / rect.width,
    scaleY = canvas.clientHeight / rect.height;


  const pos = {
    // x: (event.clientX - rect.left) ,
    // y: (event.clientY - rect.top) 
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };

  console.log(pos.x + "\n" + pos.y)

  const context = canvas.getContext("2d");
  context.fillStyle = "#FF3030";
  if (points.length < 4) {
    context.beginPath();
    context.arc(pos.x, pos.y, 3, 0, 2 * Math.PI);
    context.fill();
    points.push(pos);
  } else {
    puncte = [];
    points.forEach((p) => {
      puncte.push({
        x: Math.abs(p.x - pos.x),
        y: Math.abs(p.y - pos.y),
      });
    });
    punctApropiat = points.reduce((prev, curr) => {
      return Math.abs(curr.x - pos.x) + Math.abs(curr.y - pos.y) <
        Math.abs(prev.x - pos.x) + Math.abs(prev.y - pos.y)
        ? curr
        : prev;
    });

    context.clearRect(punctApropiat.x - 5, punctApropiat.y - 5, 10, 10);
    points.splice(points.indexOf(punctApropiat), 1);
    context.beginPath();
    context.arc(pos.x, pos.y, 4, 0, 2 * Math.PI);
    context.fill();
    points.push(pos);
  }
};

const button = document.getElementById("btn-clear");
// button.onclick = () => {
//   points = [];
//   context.clearRect(0, 0, canvas.width, canvas.height);
// }

// const buttonFEN = document.getElementById("btn-fen");
// buttonFEN.onclick = () => {
//   if (!esteImagineaIncarcata()) {
//     spop({
//       template: 'Trebuie incarcata o imagine!',
//       style: 'error',
//       group: 'img_incarcata',
//       autoclose: 2500
//     })
//     return;
//   }
//   if (points.length != 4) {
//     spop({
//       template: "Cele 4 colturi nu au fost selectate",
//       style: 'error',
//       group: 'puncte',
//       autoclose: 2500
//     })
//   }
// }

