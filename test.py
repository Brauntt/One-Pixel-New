import numpy as np
import pandas as pd
import matplotlib
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
from test_functions import perturb_image,predict_classes, attack_success


from networks.lenet import LeNet
from networks.resnet import ResNet

from differential_evolution import differential_evolution
import helper

matplotlib.style.use('ggplot')
np.random.seed(100)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
image_id = 99
helper.plot_image(x_test[image_id])

pixel = np.array([16,16,255,255,0])

image_perturbed = perturb_image(pixel, x_test[image_id])[0]

helper.plot_image(image_perturbed)

lenet = LeNet()
resnet = ResNet()
model = resnet
models = [lenet]

network_stats, correct_imgs = helper.evaluate_models(models, x_test, y_test)
correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

network_stats

true_class = y_test[image_id,0]
prior_confidence = model.predict_one(x_test[image_id])[true_class]
confidence = predict_classes(pixel, x_test[image_id], true_class, model)[0]
success = attack_success(pixel, x_test[image_id], true_class, model, verbose= True)

print('Confidence in true class', class_names[true_class], 'is', confidence)
print('Prior confidence was', prior_confidence)
print('Attack Success:', success == True)
helper.plot_image(perturb_image(pixel, x_test[image_id])[0])


def attack(img_id, model, target=None, pixel_count=1,
           maxiter=75, popsize=400, verbose=False):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else y_test[img_id, 0]

    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    bounds = [(0, 32), (0, 32), (0, 256), (0, 256), (0, 256)] * pixel_count

    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))

    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return predict_classes(xs, x_test[img_id], target_class,
                               model, target is None)

    def callback_fn(x, convergence):
        return attack_success(x, x_test[img_id], target_class,
                              model, targeted_attack, verbose)

    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_image = perturb_image(attack_result.x, x_test[img_id])[0]
    prior_probs = model.predict_one(x_test[img_id])
    predicted_probs = model.predict_one(attack_image)
    predicted_class = np.argmax(predicted_probs)
    actual_class = y_test[img_id, 0]
    success = predicted_class != actual_class
    cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

    # Show the best attempt at a solution (successful or not)
    helper.plot_image(attack_image, actual_class, class_names, predicted_class)

    return [model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
            predicted_probs, attack_result.x]

image_id_untar = 102
pixels = 1
untar_image = attack(image_id_untar, model, pixel_count = 1, verbose= True)

image_id_tar = 108
pixels_tar = 3
target_class = 1
model_tar = lenet
print('Attacking with target', class_names[target_class])
tar_image = attack(image_id_tar, model_tar, target_class, pixel_count=pixels_tar, verbose=True)


def attack_all(models, samples=500, pixels=(1, 3), targeted=False,
               maxiter=75, popsize=400, verbose=False):
    results = []
    for model in models:
        model_results = []
        valid_imgs = correct_imgs[correct_imgs.name == model.name].img
        img_samples = np.random.choice(valid_imgs, samples, replace=False)

        for pixel_count in pixels:
            for i, img_id in enumerate(img_samples):
                print('\n', model.name, '- image', img_id, '-', i + 1, '/', len(img_samples))
                targets = [None] if not targeted else range(10)

                for target in targets:
                    if targeted:
                        print('Attacking with target', class_names[target])
                        if target == y_test[img_id, 0]:
                            continue
                    result = attack(img_id, model, target, pixel_count,
                                    maxiter=maxiter, popsize=popsize,
                                    verbose=verbose)
                    model_results.append(result)

        results += model_results
        helper.checkpoint(results, targeted)
    return results

untargeted = attack_all(models, samples=3, targeted=False)

targeted = attack_all(models, samples=3, targeted=True)



untargeted, targeted = helper.load_results()

columns = ['model', 'pixels', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs', 'perturbation']
untargeted_results = pd.DataFrame(untargeted, columns=columns)
targeted_results = pd.DataFrame(targeted, columns=columns)


helper.attack_stats(untargeted_results, models, network_stats)


helper.attack_stats(targeted_results, models, network_stats)


print('Untargeted Attack')
helper.visualize_attack(untargeted_results, class_names)

#%%

print('Targeted Attack')
helper.visualize_attack(targeted_results, class_names)