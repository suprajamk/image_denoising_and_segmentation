import random
import numpy as np
from functions.io_data import read_data, write_data

class IsingModel:
    def __init__(self, image, ext_factor, beta):

        self.width, self.height, self.ext_factor, self.beta = image.shape[0], image.shape[1], ext_factor, beta
        self.image = image

    def neighbours(self, x, y):
        n = []
        if x == 0:
            n.append((self.width-1, y))
        else:
            n.append((x-1, y))
        if x == self.width-1:
            n.append((0, y))
        else:
            n.append((x+1, y))
        if y == 0:
            n.append((x, self.height-1))
        else:
            n.append((x, y-1))
        if y == self.height-1:
            n.append((x, 0))
        else:
            n.append((x, y+1))
        return n

    def local_energy(self, x, y):
        return self.ext_factor[x,y] + sum(self.image[xx,yy] for (xx, yy) in self.neighbours(x, y))

    def gibbs_sample(self, x, y):
        p = 1 / (1 + np.exp(-2 * self.beta * self.local_energy(x,y)))
        if random.uniform(0, 1) <= p:
            self.image[x, y] = 1
        else:
            self.image[x, y] = -1


def denoise(image, q, burn_in, iterations):
    external_factor = 0.5 * np.log(q / (1-q))
    model = IsingModel(image, external_factor*image, 3)

    avg = np.zeros_like(image).astype(np.float64)
    for i in range(burn_in + iterations):
        print("Iteration - " + str(i))
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if(random.uniform(0, 1) <= 0.7):
                    model.gibbs_sample(x, y)
        if(i > burn_in):
            avg += model.image
    return avg / iterations

def main():

    for img in range(2,3):
        print("Denoising for image " + str(img))
        data, image = read_data("../a1/"+str(img)+"_noise.txt", True)

        print(data.shape)
        print(image.shape)

        image[image == 0] = -1
        image[image == 255] = 1

        avg = denoise(image, 0.7, 5, 10)

        avg[avg >= 0] = 255
        avg[avg < 0] = 0

        print(avg.shape)
        width = avg.shape[0]
        height = avg.shape[1]
        counter = 0

        for i in range(0, width):
            for j in range(0, height):
                data[counter][2] = avg[i][j][0]
                counter = counter + 1

        write_data(data, "../output/"+str(img)+"_denoise.txt")
        read_data("../output/"+str(img)+"_denoise.txt", True, save=True, save_name="../output/"+str(img)+"_denoise.jpg")
        print("Finished writing data. Please check "+str(img)+"_denoise.jpg \n")

if __name__ == "__main__":
    main()
