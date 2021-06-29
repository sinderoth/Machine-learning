import sys
import kmeans
import numpy as np
from PIL import Image

def main():
    # reading pixels
    print("Reading pixels...", end="\t")
    
    # read image pixels
    # and have a list in 1 X (width * height) dimensions
    args = sys.argv[1:]
    im = Image.open(args[0],"r") 
    inp = list(im.getdata())
    inp = np.asarray(inp)
    number_of_colors= int (args[1])
    max_iterations= int (args[2])
    epsilon=float(args[3])
    distance_metric="euclidian"
    
    print("DONE")
    
    model = kmeans.KMeans(
        X=np.array(inp),
        n_clusters=number_of_colors,
        max_iterations=max_iterations,
        epsilon=epsilon,
        distance_metric="euclidian"
    )
    print("Fitting...")
    model.fit()    
    print("Fitting... DONE")

    print("Predicting...")
    color1 = (134, 66, 176)
    color2 = (34, 36, 255)
    color3 = (94, 166, 126)
    print(f"Prediction for {color1} is cluster {model.predict(color1)}")
    print(f"Prediction for {color2} is cluster {model.predict(color2)}")
    print(f"Prediction for {color3} is cluster {model.predict(color3)}")

    # replace image pixels with color palette
    # (cluster centers) found in the model

   
    wid, hei = im.size
    for x in range(wid):
        for y in range(hei):
            a = model.predict(im.getpixel((x,y)))
            newcol = model.cluster_centers[a]
            intcol=[0,0,0]
            intcol[0] = int (newcol[0])
            intcol[1] = int (newcol[1])
            intcol[2] = int (newcol[2])
            intcol = tuple(intcol)
            im.putpixel((x,y),intcol)
    
    im.save("new.jpg")


if __name__ == "__main__":
    main()