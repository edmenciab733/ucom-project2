# ucom-project2

Video de Presentacion del funcionamiento del proyecto: 

https://www.loom.com/share/ec2c9dea09e4484599cefc200b0f6656


Modelado de Keypoints

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/edmenciab733/4bea58716ee3e9defa70f86be86b437c/project1.ipynb)

Errores casuales en la aplicacion:
Linea 260: could not broadcast input array from shape (25,25,3) into shape (11,25,3)

```python
res[int(keypoint[index_mounth]):int(keypoint[index_mounth])+size_mouth, int(keypoint[index_mounth -1]):int(keypoint[index_mounth -1])+size_mouth ] = mouth
## Solucion(mala): 
##aux = res[int(keypoint[index_mounth]):int(keypoint[index_mounth])+size, int(keypoint[index_mounth -1]):int(keypoint[index_mounth -1])+size ]
## mouth = cv2.resize(mouth,(int(aux.shape[1]),int(aux.shape[0])))
```


Ayuda:

La tecla A activa la funcion de mostrar la boca mas grande(Puño cerrado o dedo de ok )

La tecla D muestra la lengua y el ojo resaltado(Mano Abierta)

La tecla S muestra los keypoints encontrados por el modelo (Movimiento desconocido en el cuadro)

La tecla L  desactiva la funcion  del teclado y deja a cargo del model  y area en verde para la deteccion del gesto. 

La tecla Q es para salir.

------


Generalidades: 

- El modelo de gestos es malo aun, creo que mas que por el modelo es por los datos, para reconocer otras manos que no sean mias ya que fueron entradas solo con dos mil imagenes todas sacadas en el mismo contexto.

- Las imagenes son sacadas de internet. 

Deteccion de gestos: Con un foto blanco, el filtro por HSV detecta una gama de colores y en base a esa gama de colores envia al modelo el cuadro marcado en verde

```python
lower_color = np.array([0,20,70], dtype=np.uint8)
upper_color = np.array([20,255,255], dtype=np.uint8)
roi=resultImage[50:500, 50:500]
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_color, upper_color)
cv2.rectangle(resultImage,(50,50),(500,500),(0,255,0),0)

```

