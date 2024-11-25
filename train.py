raw_folder = "path"
def save_data(raw_folder=raw_folder):

    dest_size = (224, 224)
    print("Bắt đầu xử lý ảnh...")

    pixels = []
    labels = []

    for folder in listdir(raw_folder):
        if folder!='.DS_Store':
            print("Folder=",folder)
            for file in listdir(raw_folder  + folder):
                if file!='.DS_Store':
                    # print("File=", file)
                    pixels.append( cv2.resize(cv2.imread(raw_folder  + folder +"/" + file),dsize=(224,224)))
                    labels.append( folder)

    pixels = np.array(pixels)
    labels = np.array(labels)

    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print(labels)

    file = open('pix.data', 'wb')
    pickle.dump((pixels,labels), file)
    file.close()

    return

save_data()

def load_data():
    file = open('/kaggle/working/pix.data', 'rb')

    # dump information to that file
    (pixels, labels) = pickle.load(file)

    # close the file
    file.close()

    print(pixels.shape)
    print(labels.shape)


    return pixels, labels


X,y = load_data()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=100)

print(X_train.shape)
print(y_train.shape)

def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Tao model
    input = Input(shape=(224, 224, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    # Them cac layer FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(7, activation='softmax', name='predictions')(x)


    
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', 
                 optimizer=Adam(learning_rate=1e-4), 
                 metrics=['accuracy'])

    return my_model

vggmodel = get_model()

filepath="weights-{epoch:02d}-{val_accuracy:.2f}.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

aug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    rescale=1./255,
    fill_mode="nearest"
)


aug_val = ImageDataGenerator(rescale=1./255)

vgghist=vggmodel.fit(aug.flow(X_train, y_train, batch_size=128),
                               epochs=50,
                               validation_data=aug.flow(X_test,y_test,
                               batch_size=128
                               ),
                     callbacks = callbacks_list
                    )

vggmodel.save("vggmodel.h5")