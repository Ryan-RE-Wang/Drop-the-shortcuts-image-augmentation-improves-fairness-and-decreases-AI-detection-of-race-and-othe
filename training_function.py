from Custom_losses import *
import time
import IPython.display
import numpy as np
import tensorflow as tf


epochs = 15

BATCH_SIZE = 128

def train_step(model, algo, mode, checkpoint_filepath, optimizer, manager, train_dataset, val_dataset):
    
    best_val_loss = np.Inf
    early_stopping_count = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.

        if (algo == 'FairALM'):
            lag_mult_r = tf.zeros(len(group_type['race']) * len(surrogate_fns))
            lag_mult_g = tf.zeros(len(group_type['gender']) * len(surrogate_fns))
            lag_mult_a = tf.zeros(len(group_type['age']) * len(surrogate_fns))
        else:
            pass

        for step, (x_batch_train, y_batch_train, demo_batch_train) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                tape.watch(model.trainable_variables)

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                if (algo == 'ERM'):
                    loss_value = ERM_Loss(y_batch_train, logits)
                elif (algo == 'Adv'):
                    loss_value_race = reciprocal_CCE_loss(demo_batch_train[:, 0], logits[0])
                    loss_value_gender = reciprocal_CCE_loss(demo_batch_train[:, 1], logits[1])
                    loss_value_age = reciprocal_CCE_loss(demo_batch_train[:, 2], logits[2])
                    loss_value_disease = ERM_Loss(y_batch_train, logits[3])
                    loss_value = loss_value_race + loss_value_gender + loss_value_age + loss_value_disease
                elif (algo == 'DistMatch'):
                    loss_value, penalty_r = DistMatch_Loss(y_batch_train, logits, demo_batch_train[:, 0], mode, 'race')
                    loss_value, penalty_g = DistMatch_Loss(y_batch_train, logits, demo_batch_train[:, 1], mode, 'gender')
                    loss_value, penalty_a = DistMatch_Loss(y_batch_train, logits, demo_batch_train[:, 2], mode, 'age')
                    loss_value = (loss_value + penalty_r + penalty_g + penalty_a)
                elif (algo == 'FairALM'):
                    loss_value, penalty_r, lag_mult_r = fairALM_loss(y_batch_train, logits, demo_batch_train[:, 0], lag_mult_r, 'race', False)
                    loss_value, penalty_g, lag_mult_g = fairALM_loss(y_batch_train, logits, demo_batch_train[:, 1], lag_mult_g, 'gender', False)
                    loss_value, penalty_a, lag_mult_a = fairALM_loss(y_batch_train, logits, demo_batch_train[:, 2], lag_mult_a, 'age', False)
                    loss_value += (loss_value + penalty_r + penalty_g + penalty_a)
                else:
                    print('Wrong algo!')
                    break

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))
                
        loss_value_val = 0    
        for x_batch_val, y_batch_val, demo_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)

            if (algo == 'ERM'):
                loss_value_val = ERM_Loss(y_batch_val, val_logits)
            elif (algo == 'Adv'):
                loss_value_race = reciprocal_CCE_loss(demo_batch_val[:, 0], val_logits[0])
                loss_value_gender = reciprocal_CCE_loss(demo_batch_val[:, 1], val_logits[1])
                loss_value_age = reciprocal_CCE_loss(demo_batch_val[:, 2], val_logits[2])
                loss_value_disease = ERM_Loss(y_batch_val, val_logits[3])
                loss_value_val = loss_value_race + loss_value_gender + loss_value_age + loss_value_disease
            elif (algo == 'DistMatch'):
                loss_value_val, penalty_r = DistMatch_Loss(y_batch_val, val_logits, demo_batch_val[:, 0], mode, 'race')
                loss_value_val, penalty_g = DistMatch_Loss(y_batch_val, val_logits, demo_batch_val[:, 1], mode, 'gender')
                loss_value_val, penalty_a = DistMatch_Loss(y_batch_val, val_logits, demo_batch_val[:, 2], mode, 'age')
                loss_value_val = (loss_value_val + penalty_r + penalty_g + penalty_a)
            elif (algo == 'FairALM'):
                loss_value_val, penalty_r, lag_mult_r = fairALM_loss(y_batch_val, val_logits, demo_batch_val[:, 0], lag_mult_r, 'race', False)
                loss_value_val, penalty_g, lag_mult_g = fairALM_loss(y_batch_val, val_logits, demo_batch_val[:, 1], lag_mult_g, 'gender', False)
                loss_value_val, penalty_a, lag_mult_a = fairALM_loss(y_batch_val, val_logits, demo_batch_val[:, 2], lag_mult_a, 'age', False)
                loss_value_val += (loss_value_val + penalty_r + penalty_g + penalty_a)
            else:
                print('Wrong algo!')
                break
                
        if (loss_value_val < best_val_loss):
            best_val_loss = loss_value_val
            save_path = manager.save()
            print("Saved checkpoint: {}".format(save_path))

            early_stopping_count = 0
        else:
            early_stopping_count += 1
            

        print("Validation loss: %.4f" % (float(loss_value_val),))
        print("Time taken: %.2fs" % (time.time() - start_time))

        if (early_stopping_count >= 4):
            print("Early stopping!")
            break

        if epoch % 2 == 0:
            optimizer.learning_rate = optimizer.learning_rate * tf.math.exp(-0.05)
            
    IPython.display.clear_output()
    