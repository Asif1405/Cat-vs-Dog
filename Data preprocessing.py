#importing data
data = "PetImages"

#defining categories
CATEGORIES = ["Dog", "Cat"]

#creating array
for category in CATEGORIES:  
    path = os.path.join(data,category)  
    for img in os.listdir(path): 
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  
        plt.show()  

print(img_array)

IMG_SIZE = 128
new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

#creating training data
training_data = []

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(data,category)
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num]) 
            except Exception as e:  
                pass

create_training_data()

print(len(training_data)) #how many training data available

random.shuffle(training_data)

#create x and y
x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#saving x
pickle_out = open("x.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

#saving y
pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
