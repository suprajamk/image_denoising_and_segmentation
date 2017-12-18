import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from functions.io_data import read_data, write_data

# Given: K = 2, the number of clusters is 2.
# Init mean1,covariance1,mixingcoeff1 and mean2,covariance2,mixingcoeff2

def fit_EM(m1, m2, filename, data, image):
    mean1 = m1              #Mean 1

    mean2 = m2              #Mean 2

    eps = 0.5               #Threshold

    z = np.asarray([[13, 20, 29], [13, 23, 37], [13, 23, 29]])
    cov1 = np.cov(z)        #covariance 1

    z = np.asarray([[9, -58, 7], [8, -7, 10], [6, -4, 6]])
    cov2 = np.cov(z)        #covariance 2

    mix1 = 0.4              #mixing co-efficient 1
    mix2 = 0.6              #mixing co-efficient 2

    N = image.shape[0] * image.shape[1]         #Total number of samples

    log_likelihoods = []

    print("Initialization of mean,covariance and mixing co-efficient statement done for "+str(filename))

    print("")
    print("Starting EM algorithm for "+str(filename))
    # Expectation Step :

    iteration_number = 0
    while (1):

        iteration_number += 1
        print("Iteration: "+str(iteration_number))

        N1 = 0
        N2 = 0
        resp1_list = []
        resp2_list = []
        mu_sum1 = [0, 0, 0]
        mu_sum2 = [0, 0, 0]

        for y in image:
            for x in y:
                prob1 = multivariate_normal.pdf(x, mean=mean1, cov=cov1, allow_singular=True)      # gaussian density 1

                prob2 = multivariate_normal.pdf(x, mean=mean2, cov=cov2, allow_singular=True)      # gaussian density 2

                Numerator1 = mix1 * prob1
                Numerator2 = mix2 * prob2

                denom = Numerator1 + Numerator2

                resp1 = Numerator1 / denom  #responsibility for 1st cluster

                resp2 = Numerator2 / denom  #responsibility for 2nd cluster

                resp1_list.append(resp1)
                resp2_list.append(resp2)


                mu_sum1 += resp1 * x
                mu_sum2 += resp2 * x

                N1 += resp1
                N2 += resp2

        # Maximization Step :

        mu_new1 = mu_sum1 / N1  #updated mean 1
        mu_new2 = mu_sum2 / N2  #updated mean 2

        var_1 = np.zeros((3, 3))
        var_2 = np.zeros((3, 3))

        i = 0
        for y in image:
            for x in y:
                var_1 += resp1_list[i] * np.outer((x - mu_new1), (x - mu_new1))
                var_2 += resp2_list[i] * np.outer((x - mu_new2), (x - mu_new2))
                i = i + 1

        var_new1 = var_1 / N1   #updated covariance1
        var_new2 = var_2 / N2   #updated covariance2

        mix_new1 = N1 / N       #updated mixing co-efficient1
        mix_new2 = N2 / N       #updated mixing co-efficient2

        mean1 = mu_new1
        mean2 = mu_new2

        cov1 = var_new1
        cov2 = var_new2

        mix1 = mix_new1
        mix2 = mix_new2

        #Calculate Log Likelihood
        Z = [0, 0]
        ll = 0
        sumList=[]
        for y in image:
            for x in y:
                prob1 = multivariate_normal.pdf(x, mu_new1, var_new1, allow_singular=True)

                prob2 = multivariate_normal.pdf(x, mu_new2, var_new2, allow_singular=True)

                sum = (mix_new1 * prob1) + (mix_new2 * prob2)
                sumList.append(np.log(sum))

            ll = np.sum(np.asarray(sumList))


        log_likelihoods.append(ll)

        print("Log Likelihood: " + str(ll))

        if len(log_likelihoods) < 2: continue
        if np.abs(ll - log_likelihoods[-2]) < eps: break
        #Break loop if log likelihoods dont change more than threshold over 2 iterations

    print("")
    print("End of iterations for: " + str(filename))
    print("")

    #Write to File
    print("Writing to file for:  " + str(filename))

    back_data = data.copy()
    front_data = data.copy()
    mask_data = data.copy()

    for i in range(0,len(data)-1):

        cell = data[i]
        point = [cell[2], cell[3], cell[4]]
        prob1 = multivariate_normal.pdf(point, mean=mean1, cov=cov1, allow_singular=True)

        resp1 = mix1 * prob1
        prob2 = multivariate_normal.pdf(point, mean=mean2, cov=cov2, allow_singular=True)
        resp2 = mix2 * prob2

        resp1 = resp1/(resp1+resp2)
        resp2 = resp2/(resp1+resp2)


        if (resp1 < resp2):
            back_data[i][2] = back_data[i][3] = back_data[i][4] = 0
            mask_data[i][2] = mask_data[i][3] = mask_data[i][4] = 0

        else:
            front_data[i][2] = front_data[i][3] = front_data[i][4] = 0
            mask_data[i][2] = 100
            mask_data[i][3] = mask_data[i][4] = 0


    write_data(back_data,"../output/"+str(filename)+"_back.txt")
    read_data(str(filename)+"_back.txt", False, save=True, save_name="../output/"+str(filename)+"_background.jpg")


    write_data(front_data,"../output/"+str(filename)+"_fore.txt")
    read_data(str(filename)+"_fore.txt", False, save=True, save_name="../output/"+str(filename)+"_foreground.jpg")


    write_data(mask_data,"../output/"+str(filename)+"_mask.txt")
    read_data(str(filename)+"_mask.txt", False, save=True, save_name="../output/"+str(filename)+"_masked.jpg")

    print("Finished writing data. Please check "+str(filename)+"_background.jpg, "+str(filename)+
          "_foreground.jpg and "+str(filename)+"_masked.jpg ")

def main():

    #Approximate mean values obtained by running K-means using scikit library

    data, image = read_data("../a2/cow.txt", True)
    X = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    clusters = kmeans.cluster_centers_
    fit_EM(clusters[0],clusters[1],"cow",data,image)

    data, image = read_data("../a2/fox.txt", True)
    X = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    clusters = kmeans.cluster_centers_
    fit_EM(clusters[0],clusters[1], "fox", data, image)

    data, image = read_data("../a2/owl.txt", True)
    X = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    clusters = kmeans.cluster_centers_
    fit_EM(clusters[0], clusters[1],"owl",data,image)

    data, image = read_data("../a2/zebra.txt", True)
    X = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    clusters = kmeans.cluster_centers_
    fit_EM(clusters[0], clusters[1], "zebra", data, image)

if __name__ == "__main__":
    main()