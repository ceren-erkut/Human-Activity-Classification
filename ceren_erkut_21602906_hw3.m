function ceren_erkut_21602906_hw3(question)
clc
close all
switch question
    case '1'
        disp('1')
        %% QUESTION 1 PART A
        disp('=== Question 1 Part A solution is initiated. ===')
        tic
        dataset = h5read('assign3_data1.h5','/data');
        gray_scale_samples = zeros(16,16,length(dataset));
        for i=1:length(dataset)
            gray_scale_samples(:,:,i) = dataset(:,:,1,i)*0.2126 + dataset(:,:,2,i)*0.7152 + dataset(:,:,3,i)*0.0722; % convert to grayscale
            gray_scale_samples(:,:,i) = gray_scale_samples(:,:,i) - mean(mean(gray_scale_samples(:,:,i))); % subtract the mean
        end
        
        gray_scale_std_value = std(reshape(gray_scale_samples, 16*16*length(dataset),1));
        for k = 1:length(gray_scale_samples)
            sample_image = gray_scale_samples(:,:,k);
            % clip around +/- 3
            sample_image(sample_image >= 3*gray_scale_std_value) = 3*gray_scale_std_value;
            sample_image(sample_image <= -3*gray_scale_std_value) = -3*gray_scale_std_value;
            % map samples
            gray_scale_samples(:,:,k) = sample_image*(0.8 / (6*gray_scale_std_value));
        end
        gray_scale_samples = gray_scale_samples + 0.5;
        random_index = randi([1, length(dataset)], 200,1);
        
        % display random 200 samples
        figure
        for i=1:200
            subplot(20,10,i)
            imshow(dataset(:,:,:,random_index(i)))
        end
        figure
        for i=1:200
            subplot(20,10,i)
            imshow(gray_scale_samples(:,:,random_index(i)))
        end
        batch_input = zeros(16*16, length(dataset));
        for i = 1:length(gray_scale_samples)
            batch_input(:,i) = reshape(gray_scale_samples(:,:,i), 1, 16*16);
        end
        elapsedTime = toc/60;
        disp("Time passed in Question 1 Part A : " + elapsedTime + " min");
        
        %% QUESTION 1 PART B
        disp('=== Question 1 Part B solution is initiated. ===')
        tic
        L_in = 256;
        L_hid = 96;
        L_out = 256;
        rho = 0.4;
        lambda = 2 * 10^-4;
        beta = 0.0015;
        params = [L_in, L_hid, lambda, rho, beta];
        
        %input to hidden
        w_o = sqrt(6 / (L_in + L_hid));
        w_ih = -w_o + rand(L_hid, L_in)*2*w_o;
        b_ih = -w_o + rand(L_hid, 1)*2*w_o;
        %hidden to output
        w_ho = w_ih';
        w_o = sqrt(6 / (L_hid + L_out));
        b_ho = -w_o + rand(L_out, 1)*2*w_o;
        
        learning_rate = 0.3;
        epoch_num = 2000;
        
        epoch_error = zeros(1, epoch_num);
        
        for epoch = 1:epoch_num
            
            [w_ih, w_ho, b_ih, b_ho, epoch_error] = epoch_training(params, w_ih, w_ho, b_ih, b_ho, learning_rate, batch_input, epoch_error, epoch);
            
        end
        
        % test with trained network parameters
        v_hidden_layer = w_ih * batch_input + b_ih;
        o_hidden_layer = 1./(1 + exp(-v_hidden_layer));
        v_output_layer = w_ho * o_hidden_layer + b_ho;
        o_output_layer = 1./(1 + exp(-v_output_layer));
        
        % plot original and output images
        figure
        sgtitle("Comparisons with 2 Image Samples | " + "Rho: " + params(4) + ", Beta: " + params(5) + ", # of Hidden Neurons: " + params(2) + ", lambda: " + params(3));
        rgb_images = dataset(:,:,:,7329);
        subplot(3,3,1);
        imshow(rgb_images);
        title('Original RGB Images');
        gray_images = reshape(batch_input(:,7329), 16, 16);
        subplot(3,3,2);
        imshow(gray_images);
        title('Gray Scale Images');
        output_autoencoder = reshape(o_output_layer(:,7329), 16, 16);
        subplot(3,3,3);
        imshow(output_autoencoder);
        title('Output Images of Autoencoder');
        
        rgb_images = dataset(:,:,:,7076);
        subplot(2,3,4);
        imshow(rgb_images);
        title('Original RGB Images');
        gray_images = reshape(batch_input(:,7076), 16, 16);
        subplot(2,3,5);
        imshow(gray_images);
        title('Gray Scale Images');
        output_autoencoder = reshape(o_output_layer(:,7076), 16, 16);
        subplot(2,3,6);
        imshow(output_autoencoder);
        title('Output Images of Autoencoder');
        
        figure
        plot(epoch_error);
        title("Cost versus Epoch | " + "Rho: " + params(4) + " Beta: " + params(5) + ", # of Hidden Neurons: " + params(2) + ", lambda: " + params(3));
        elapsedTime = toc/60;
        disp("Time passed in Question 1 Part B : " + elapsedTime + " min");
        
        %% QUESTION 1 PART C & D
        disp('=== Question 1 Part C & D solution is initiated. ===')
        figure
        sgtitle("Hidden Layer Weights | " + "Rho: " + params(4) + " Beta: " + params(5) + ", # of Hidden Neurons: " + params(2) + ", lambda: " + params(3));
        num_hidden_weights = ceil(sqrt(L_hid));
        for i = 1:L_hid
            subplot(num_hidden_weights, num_hidden_weights, i);
            picture = reshape(w_ih(i,:), 16, 16);
            imshow(picture);
        end
        
        
    case '3'
        disp('3')
        
        trX = h5read('assign3_data3.h5','/trX');
        tstX = h5read('assign3_data3.h5','/tstX');
        trY = h5read('assign3_data3.h5','/trY');
        tstY = h5read('assign3_data3.h5','/tstY');
        
        input_size = 3;
        output_size = 6;
        hidden_size = 128;
        
        epoch_num = 50;
        batch_size = 32;
        learning_rate = 0.01;
        alpha = 0.65;
        class_num = 6;
        
        epoch_error_training = zeros(1, epoch_num);
        epoch_error_validation = zeros(1, epoch_num);
        
        % initialization
        w0_ih = sqrt(6/(input_size+hidden_size));
        w0_ho = sqrt(6/(hidden_size+output_size));
        w0_hh = sqrt(6/(2*hidden_size));
        
        %% QUESTION 3 PART A
        %tic
        disp('=== Question 3 Part A solution is initiated. ===')
        w_ho = -w0_ho + 2 * w0_ho * rand(output_size, hidden_size);
        w_hh = -w0_hh + 2* w0_hh * rand(hidden_size, hidden_size);
        w_ih = -w0_ih + 2 * w0_ih * rand(hidden_size, input_size);
        
        b_hidden = -w0_hh + 2 * w0_hh * rand(hidden_size, 1);
        b_output = -w0_ho + 2 * w0_ho * rand(output_size, 1);
        
        alpha_hh = zeros(hidden_size, hidden_size);
        alpha_ih = zeros( hidden_size, input_size);
        alpha_ho = zeros(output_size, hidden_size);
        alpha_bh = zeros(hidden_size, 1);
        alpha_bo = zeros(output_size, 1);
        
        updated_hh = zeros(hidden_size, hidden_size);
        updated_ih = zeros(hidden_size, input_size);
        updated_ho = zeros(output_size, hidden_size);
        updated_b_h = zeros(hidden_size, 1);
        updated_b_o = zeros(output_size, 1);
        
        create validation set
        validation_set = zeros(3,150,300);
        validation_set_label = zeros(class_num,300);
        training_set = zeros(3,150,2700);
        training_set_label = zeros(class_num,2700);
        random_index = randperm(500);
        
        for i = 1:class_num
            validation_set(:, :, (i-1)*50+1 : i*50) = trX(:, :, (i-1)*500+random_index(1:50));
            validation_set_label(:,(i-1)*50+1:i*50) = trY(:,(i-1)*500+random_index(1:50));
            training_set(:,:,(i-1)*450+1:i*450) = trX(:,:,(i-1)*500+random_index(51:500));
            training_set_label(:,(i-1)*450+1:i*450) = trY(:,(i-1)*500+random_index(51:500));
        end
        
        % iterate through epochs
        for epoch = 1:epoch_num
            
            rnd_sample_index = randperm(2700);
            count = 0;
            epoch_loss = 0;
            cci_training = 0;
            confusion_matrix_test = zeros(class_num, class_num);
            confusion_matrix_train = zeros(class_num, class_num);
            
            for index = 1:2700
                
                sample_data = training_set(:, :, rnd_sample_index(index));
                sample_data_label = training_set_label(:, rnd_sample_index(index));
                
                % single layer forward pass
                output = zeros(length(w_hh), 1);
                output_store = zeros(length(w_hh), 1, 150);
                for i = 1:150
                    potential = w_ih * sample_data(:,i) + w_hh * output + b_hidden;
                    output = tanh(potential);
                    output_store(:,:,i+1) = output;
                end
                o = w_ho * output + b_output;
                
                % softmax layer
                o = o - max(o);
                o = exp(o) ./ sum(exp(o));
                
                [~, label] = max(o);
                [~, true_label] = max(sample_data_label);
                
                confusion_matrix_train(label, true_label) = confusion_matrix_train(label, true_label) + 1;
                if (label == true_label)
                    cci_training = cci_training+1;
                end
                
                loss_gradient = -(sample_data_label' * log(o));
                
                epoch_loss = epoch_loss + loss_gradient;
                
                d_y = -sample_data_label+o;
                
                % backward pass RNN
                [del_w_ih , del_w_hh, del_w_ho, del_bias_hidden, del_bias_output] = bptt(sample_data, w_ho, w_hh, d_y, output_store);
                updated_hh = updated_hh + del_w_hh;
                updated_ih = updated_ih + del_w_ih;
                updated_ho = updated_ho + del_w_ho;
                updated_b_h = updated_b_h + del_bias_hidden;
                updated_b_o = updated_b_o + del_bias_output;
                count = count +1;
                
                if count == batch_size
                    count = 0;
                    alpha_hh = alpha_hh*alpha + learning_rate * (1/batch_size) * (updated_hh);
                    w_hh = w_hh - alpha_hh;
                    alpha_ih = alpha_ih*alpha + learning_rate * (1/batch_size)* (updated_ih);
                    w_ih = w_ih -  alpha_ih;
                    alpha_ho = alpha_ho*alpha + learning_rate * (1/batch_size)* (updated_ho);
                    w_ho = w_ho - alpha_ho;
                    alpha_bh = alpha_bh*alpha + learning_rate * (1/batch_size)* (updated_b_h);
                    b_hidden = b_hidden - alpha_bh;
                    alpha_bo = alpha_bo*alpha + learning_rate * (1/batch_size)* (updated_b_o);
                    b_output = b_output - alpha_bo;
                    updated_hh = zeros(hidden_size, hidden_size);
                    updated_ih = zeros(hidden_size, input_size);
                    updated_ho = zeros(output_size, hidden_size);
                    updated_b_h = zeros(hidden_size, 1);
                    updated_b_o = zeros(output_size, 1);
                    
                end
                
                if index == 2700
                    count = 0;
                    batch_size = 12;
                    alpha_hh = alpha_hh*alpha + learning_rate * (1/batch_size) * (updated_hh);
                    w_hh = w_hh - alpha_hh;
                    alpha_ih = alpha_ih*alpha + learning_rate * (1/batch_size)* (updated_ih);
                    w_ih = w_ih -  alpha_ih;
                    alpha_ho = alpha_ho*alpha + learning_rate * (1/batch_size)* (updated_ho);
                    w_ho = w_ho - alpha_ho;
                    alpha_bh = alpha_bh*alpha + learning_rate * (1/batch_size)* (updated_b_h);
                    b_hidden = b_hidden - alpha_bh;
                    alpha_bo = alpha_bo*alpha + learning_rate * (1/batch_size)* (updated_b_o);
                    b_output = b_output - alpha_bo;
                    
                    updated_hh = zeros(128,128);
                    updated_ih = zeros(128,3);
                    updated_ho = zeros(6,128);
                    updated_b_h = zeros(128,1);
                    updated_b_o = zeros(6,1);
                    
                    batch_size = 32;
                end
            end
            
            
            epoch_error_training(epoch) = epoch_loss/2700;
            epoch_loss = 0;
            
            % Validation Loss and Accuracy
            validation_loss = 0;
            for m = 1:300
                
                sample_data = validation_set(:, :, m);
                sample_data_label = validation_set_label(:, m);
                
                % forward pass
                output = zeros(length(w_hh), 1);
                output_store = zeros(length(w_hh), 1, 150);
                for i = 1:150
                    potential = w_ih * sample_data(:,i) + w_hh * output + b_hidden;
                    output = tanh(potential);
                    output_store(:,:,i+1) = output;
                end
                o = w_ho * output + b_output;
                
                % softmax layer
                o = o - max(o);
                o = exp(o) ./ sum(exp(o));
                
                v_sgd = -(sample_data_label'*log(o));
                
                validation_loss = validation_loss + v_sgd;
            end
            
            % Test Trace for Accuracy
            cci_test = 0;
            for k = 1:600
                
                sample_data = tstX(:, :, k);
                sample_data_label = tstY(:, k);
                
                % forward pass
                output = zeros(length(w_hh), 1);
                output_store = zeros(length(w_hh), 1, 150);
                for i = 1:150
                    potential = w_ih * sample_data(:,i) + w_hh * output + b_hidden;
                    output = tanh(potential);
                    output_store(:,:,i+1) = output;
                end
                o = w_ho * output + b_output;
                
                % softmax layer
                o = o - max(o);
                o = exp(o) ./ sum(exp(o));
                
                [~, label] = max(o);
                [~, true_label] = max(sample_data_label);
                
                if(label == true_label)
                    cci_test = cci_test+1;
                end
                
                confusion_matrix_test(label, true_label) =  confusion_matrix_test(label, true_label) + 1;
                
            end
            
            epoch_error_validation(epoch) = validation_loss/300;
            
            figure(20)
            plot(1:epoch , epoch_error_validation(1:epoch) ,'r');
            hold on
            plot(1:epoch, epoch_error_training(1:epoch) ,'b');
            title("Cross Entropy vs Epoch for RNN , eta = " + learning_rate + ", alpha = " + alpha)
            legend("Validation Error", "Training Error")
            xlabel("Epoch Number")
            ylabel("Error")
            grid on
            
            if validation_loss/300 < 1.22
                break;
            end
        end
        disp("Training loss: " +  epoch_error_training(epoch));
        disp("Training Accuracy: " + 100*cci_training/2700 + " %");
        disp("Training Confusion Matrix: ")
        confusion_matrix_train
        
        disp("Validation Loss: " + validation_loss/300 )
        disp("Test Accuracy: " + 100*cci_test/600 + " %");
        disp("Test Confusion Matrix: ");
        confusion_matrix_test
        elapsedTime = toc/60;
        disp("Time passed in Question 3 Part A : " + elapsedTime + " min");
        
        
        %% QUESTION 3 PART B
        tic
        disp('=== Question 3 Part B solution is initiated. ===')
        
        % initialization
        
        % Forget Gate
        w_forget = - w0_hh + 2* w0_hh * rand(hidden_size, hidden_size);
        u_forget = - w0_ih + 2 * w0_ih * rand(input_size, hidden_size);
        b_forget = 1*ones(1, hidden_size);
        % Input Gate
        w_input = - w0_hh + 2* w0_hh * rand(hidden_size, hidden_size);
        u_input = - w0_ih + 2 * w0_ih * rand(input_size, hidden_size);
        b_input = 1*ones( 1, hidden_size);
        % Output Gate
        w_output = - w0_hh + 2* w0_hh * rand(hidden_size, hidden_size);
        u_output = - w0_ih + 2 * w0_ih * rand(input_size, hidden_size);
        b_output = 1*ones(1, hidden_size);
        % Candidate State
        w_candidate = - w0_hh + 2* w0_hh * rand(hidden_size, hidden_size);
        u_candidate = - w0_ih + 2 * w0_ih * rand(input_size, hidden_size);
        b_candidate = 1*ones(1, hidden_size);
        % Final Output
        w_y_output = - w0_ho + 2 * w0_ho * rand(hidden_size, output_size);
        b_y_output = - w0_ho + 2 * w0_ho * rand(1, output_size);
        
        alpha_w_forget = zeros(hidden_size, hidden_size);
        alpha_u_forget = zeros(input_size, hidden_size);
        alpha_b_forget = zeros(1, hidden_size);
        alpha_w_input = zeros(hidden_size, hidden_size);
        alpha_u_input = zeros(input_size, hidden_size);
        alpha_b_input = zeros(1, hidden_size);
        alpha_w_output = zeros(hidden_size, hidden_size);
        alpha_u_output = zeros(input_size, hidden_size);
        alpha_b_output = zeros(1, hidden_size);
        alpha_w_g = zeros(hidden_size, hidden_size);
        alpha_u_g = zeros(input_size, hidden_size);
        alpha_b_g = zeros(1, hidden_size);
        alpha_w_y = zeros(hidden_size, output_size);
        alpha_b_y = zeros(1, output_size);
        
        % container
        w_forget_container = zeros(hidden_size, hidden_size);
        u_forget_container = zeros(input_size, hidden_size);
        b_forget_container = zeros(1 , hidden_size);
        w_input_container = zeros(hidden_size, hidden_size);
        u_input_container = zeros(input_size, hidden_size);
        b_input_container = zeros(1, hidden_size);
        w_output_container = zeros(hidden_size, hidden_size);
        u_output_container = zeros(input_size, hidden_size);
        b_output_container = zeros(1 , hidden_size);
        w_g_container = zeros(hidden_size, hidden_size);
        u_g_container = zeros(input_size, hidden_size);
        b_g_container = zeros(1, hidden_size);
        w_y_container = zeros(hidden_size, output_size);
        b_y_container = zeros(1, output_size);
        
        epoch_error_training = zeros(1, epoch_num);
        epoch_error_validation = zeros(1, epoch_num);
        % Epoch iteration
        for epoch = 1:epoch_num
            rnd_sample_index = randperm(2700);
            counter = 0;
            epoch_training_loss = 0;
            
            confusion_matrix_test = zeros(class_num, class_num);
            confusion_matrix_train = zeros(class_num, class_num);
            cci_training = 0;
            
            for k=1:2700
                sample_data = training_set(:, :, rnd_sample_index(k));
                sample_data_label = training_set_label(:, rnd_sample_index(k));
                
                % single layer forward pass
                [y, h_store, c_store, forget_store, input_store, output_store, g_store] = LSTM_forward(sample_data, w_forget, u_forget, b_forget, w_input, u_input, b_input, w_output, u_output, b_output, w_candidate, u_candidate, b_candidate, w_y_output, b_y_output);
                
                % softmax layer
                y = y - max(y);
                o = exp(y) ./ sum(exp(y));
                
                [~, label] = max(o);
                [~, true_label] = max(sample_data_label);
                
                confusion_matrix_train(label, true_label) = confusion_matrix_train(label, true_label) + 1;
                if(label == true_label)
                    cci_training = cci_training + 1;
                end
                
                v_sgd = -(log(o)*sample_data_label);
                epoch_training_loss = epoch_training_loss + v_sgd;
                
                d_y = -sample_data_label'+o;
                
                % backpropagation through time LSTM
                [d_w_y, d_b_y, del_w_forget, del_u_forget, del_b_forget, del_w_input, del_u_input, del_b_input, del_w_output, del_u_output, del_b_output, del_w_gate, del_u_g, del_b_g] = LSTM_bptt(sample_data, d_y, h_store, c_store, forget_store, input_store, output_store, g_store, w_forget, w_input, w_output, w_candidate, w_y_output, hidden_size, input_size);
                counter = counter +1;
                
                w_forget_container = w_forget_container + del_w_forget;
                u_forget_container =u_forget_container + del_u_forget;
                b_forget_container = b_forget_container + del_b_forget;
                
                w_input_container = w_input_container + del_w_input;
                u_input_container =u_input_container + del_u_input;
                b_input_container = b_input_container + del_b_input;
                
                w_output_container = w_output_container + del_w_output;
                u_output_container = u_output_container + del_u_output;
                b_output_container = b_output_container + del_b_output;
                
                w_g_container = w_g_container + del_w_gate;
                u_g_container = u_g_container + del_u_g;
                b_g_container = b_g_container + del_b_g;
                
                w_y_container = w_y_container + d_w_y;
                b_y_container = b_y_container + d_b_y;
                
                % weight update
                if counter == batch_size
                    
                    counter = 0;
                    alpha_w_forget = alpha_w_forget*alpha + learning_rate * (1/batch_size) * w_forget_container;
                    alpha_u_forget = alpha_u_forget*alpha + learning_rate * (1/batch_size) * u_forget_container;
                    alpha_b_forget = alpha_b_forget*alpha + learning_rate * (1/batch_size) * b_forget_container;
                    alpha_w_input = alpha_w_input*alpha + learning_rate * (1/batch_size) * w_input_container;
                    alpha_u_input = alpha_u_input*alpha + learning_rate * (1/batch_size) * u_input_container;
                    alpha_b_input = alpha_b_input*alpha + learning_rate * (1/batch_size) * b_input_container;
                    alpha_w_g = alpha_w_g*alpha + learning_rate * (1/batch_size) * w_g_container;
                    alpha_u_g = alpha_u_g*alpha + learning_rate * (1/batch_size) * u_g_container;
                    alpha_b_g = alpha_b_g*alpha + learning_rate * (1/batch_size) * b_g_container;
                    alpha_w_output = alpha_w_output*alpha + learning_rate * (1/batch_size) * w_output_container;
                    alpha_u_output = alpha_u_output*alpha + learning_rate * (1/batch_size) * u_output_container;
                    alpha_b_output = alpha_b_output*alpha + learning_rate * (1/batch_size) * b_output_container;
                    alpha_w_y = alpha_w_y*alpha + learning_rate * (1/batch_size) * w_y_container;
                    alpha_b_y = alpha_b_y*alpha + learning_rate * (1/batch_size) * b_y_container;
                    
                    w_forget = w_forget - alpha_w_forget;
                    u_forget = u_forget - alpha_u_forget;
                    b_forget = b_forget - alpha_b_forget;
                    w_input = w_input - alpha_w_input;
                    u_input = u_input - alpha_u_input;
                    b_input = b_input - alpha_b_input;
                    w_candidate = w_candidate - alpha_w_g;
                    u_candidate = u_candidate - alpha_u_g;
                    b_candidate = b_candidate - alpha_b_g;
                    w_output = w_output - alpha_w_output;
                    u_output = u_output - alpha_u_output;
                    b_output = b_output - alpha_b_output;
                    w_y_output = w_y_output - alpha_w_y;
                    b_y_output = b_y_output - alpha_b_y;
                    
                    w_forget_container = zeros(hidden_size, hidden_size);
                    u_forget_container = zeros(input_size, hidden_size);
                    b_forget_container = zeros(1, hidden_size);
                    w_input_container = zeros(hidden_size, hidden_size);
                    u_input_container = zeros(input_size, hidden_size);
                    b_input_container = zeros(1, hidden_size);
                    w_output_container = zeros(hidden_size, hidden_size);
                    u_output_container = zeros(input_size, hidden_size);
                    b_output_container = zeros(1, hidden_size);
                    w_g_container = zeros(hidden_size, hidden_size);
                    u_g_container = zeros(input_size, hidden_size);
                    b_g_container = zeros(1 , hidden_size);
                    w_y_container = zeros(hidden_size, output_size);
                    b_y_container = zeros(1, output_size);
                    
                end
                
                if k == 2700
                    counter = 0;
                    batch_size = 12;
                    % weight update
                    
                    alpha_w_forget = alpha_w_forget*alpha + learning_rate * (1/batch_size) * w_forget_container;
                    alpha_u_forget = alpha_u_forget*alpha + learning_rate * (1/batch_size) * u_forget_container;
                    alpha_b_forget = alpha_b_forget*alpha + learning_rate * (1/batch_size) * b_forget_container;
                    alpha_w_input = alpha_w_input*alpha + learning_rate * (1/batch_size) * w_input_container;
                    alpha_u_input = alpha_u_input*alpha + learning_rate * (1/batch_size) * u_input_container;
                    alpha_b_input = alpha_b_input*alpha + learning_rate * (1/batch_size) * b_input_container;
                    alpha_w_g = alpha_w_g*alpha + learning_rate * (1/batch_size) * w_g_container;
                    alpha_u_g = alpha_u_g*alpha + learning_rate * (1/batch_size) * u_g_container;
                    alpha_b_g = alpha_b_g*alpha + learning_rate * (1/batch_size) * b_g_container;
                    alpha_w_output = alpha_w_output*alpha + learning_rate * (1/batch_size) * w_output_container;
                    alpha_u_output = alpha_u_output*alpha + learning_rate * (1/batch_size) * u_output_container;
                    alpha_b_output = alpha_b_output*alpha + learning_rate * (1/batch_size) * b_output_container;
                    alpha_w_y = alpha_w_y*alpha + learning_rate * (1/batch_size) * w_y_container;
                    alpha_b_y = alpha_b_y*alpha + learning_rate * (1/batch_size) * b_y_container;
                    
                    w_forget = w_forget - alpha_w_forget;
                    u_forget = u_forget - alpha_u_forget;
                    b_forget = b_forget - alpha_b_forget;
                    w_input = w_input - alpha_w_input;
                    u_input = u_input - alpha_u_input;
                    b_input = b_input - alpha_b_input;
                    w_candidate = w_candidate - alpha_w_g;
                    u_candidate = u_candidate - alpha_u_g;
                    b_candidate = b_candidate - alpha_b_g;
                    w_output = w_output - alpha_w_output;
                    u_output = u_output - alpha_u_output;
                    b_output = b_output - alpha_b_output;
                    w_y_output = w_y_output - alpha_w_y;
                    b_y_output = b_y_output - alpha_b_y;
                    
                    w_forget_container = zeros(hidden_size, hidden_size);
                    u_forget_container = zeros(input_size, hidden_size);
                    b_forget_container = zeros(1, hidden_size);
                    w_input_container = zeros(hidden_size, hidden_size);
                    u_input_container = zeros(input_size, hidden_size);
                    b_input_container = zeros(1, hidden_size);
                    w_output_container = zeros(hidden_size, hidden_size);
                    u_output_container = zeros(input_size, hidden_size);
                    b_output_container = zeros(1, hidden_size);
                    w_g_container = zeros(hidden_size, hidden_size);
                    u_g_container = zeros(input_size, hidden_size);
                    b_g_container = zeros(1, hidden_size);
                    w_y_container = zeros(hidden_size, output_size);
                    b_y_container = zeros(1, output_size);
                    batch_size = 32;
                    
                end
                
            end
            
            epoch_error_training(epoch) = epoch_training_loss/2700;
            epoch_training_loss = 0;
            
            % Validation loss and accuracy
            validation_loss = 0;
            for m = 1:300
                sample_data = validation_set(:,:,m);
                sample_data_label = validation_set_label(:,m);
                % single layer forward pass
                [y, ~, ~, ~, ~, ~, ~] = LSTM_forward(sample_data, w_forget, u_forget, b_forget, w_input, u_input, b_input, w_output, u_output, b_output, w_candidate, u_candidate, b_candidate, w_y_output, b_y_output);
                % softmax layer
                y = y - max(y);
                o = exp(y) ./ sum(exp(y));
                v_sgd = -(log(o)*sample_data_label);
                validation_loss = validation_loss + v_sgd;
            end
            
            % Test accuracy
            cci_test = 0;
            for l = 1:600
                sample_data = tstX(:,:,l);
                sample_data_label = tstY(:,l);
                % single layer forward pass
                [y, ~, ~, ~, ~, ~, ~] = LSTM_forward(sample_data, w_forget, u_forget, b_forget, w_input, u_input, b_input, w_output, u_output, b_output, w_candidate, u_candidate, b_candidate, w_y_output, b_y_output);
                % softmax layer
                y = y - max(y);
                o = exp(y) ./ sum(exp(y));
                [~, label_test] = max(o);
                [~, true_test_label] = max(sample_data_label);
                
                if(label_test == true_test_label)
                    cci_test = cci_test+1;
                end
                
                confusion_matrix_test(label_test, true_test_label) = confusion_matrix_test(label_test, true_test_label) + 1;
            end
            
            epoch_error_validation(epoch) = validation_loss/300;
            
            figure(21)
            plot(1:epoch, epoch_error_validation(1:epoch), 'r');
            hold on
            plot(1:epoch, epoch_error_training(1:epoch), 'b');
            title("Cross Entropy vs Epoch for LSTM , learning rate = " + learning_rate + ", alpha = " + alpha)
            legend("Validation Error", "Training Error")
            xlabel("Epoch Number")
            ylabel("Error")
            grid on
            
            if validation_loss/300 < 1.1
                break;
            end
        end
        
        disp("Training loss: " + epoch_error_training(epoch))
        disp("Training Accuracy (%): " + 100*cci_training/2700);
        disp("Training Confusion Matrix");
        confusion_matrix_train
        
        disp("Validation loss: " + validation_loss/300);
        disp("Test Accuracy (%) :" + 100*cci_test/600);
        disp("Test Confusion Matrix");
        confusion_matrix_test
        elapsedTime = toc/60;
        disp("Time passed in Question 3 Part B : " + elapsedTime + " min");
        
        %% QUESTION 3 PART C
        tic
        disp('=== Question 3 Part C solution is initiated. ===')
        w0_y = sqrt(6/(hidden_size+output_size));
        w0_u = sqrt(6/(3+hidden_size));
        w0_w = sqrt(6/(2*hidden_size));
        
        % Input Gate
        w_reset_gate = -w0_w + 2* w0_w * rand(hidden_size, hidden_size);
        u_reset_gate = -w0_u + 2 * w0_u * rand(input_size, hidden_size);
        b_reset_gate = 2*ones(1, hidden_size);
        
        % Output Gate
        w_output_gate = -w0_w + 2* w0_w * rand(hidden_size, hidden_size);
        u_output_gate = -w0_u + 2 * w0_u * rand(input_size, hidden_size);
        b_output_gate = 2*ones(1, hidden_size);
        
        % Candidate State
        w_candidate_gate = -w0_w + 2* w0_w * rand(hidden_size, hidden_size);
        u_candidate_gate = -w0_u + 2 * w0_u * rand(input_size, hidden_size);
        b_candidate_gate = 2*ones(1, hidden_size);
        
        % Final Output
        w_y_out = - w0_y + 2 *  w0_y * rand(hidden_size, output_size);
        b_y_out = - w0_y + 2 *  w0_y * rand(1, output_size);
        
        alpha_wr = zeros(hidden_size, hidden_size);
        alpha_ur = zeros(input_size, hidden_size);
        alpha_br = zeros(1 , hidden_size);
        alpha_wz = zeros(hidden_size, hidden_size);
        alpha_uz = zeros(input_size, hidden_size);
        alpha_bz = zeros(1 , hidden_size);
        alpha_wc = zeros(hidden_size, hidden_size);
        alpha_uc = zeros(input_size, hidden_size);
        alpha_bc = zeros(1 , hidden_size);
        alpha_wy = zeros(hidden_size, output_size);
        alpha_by = zeros(1, output_size);
        
        w_reset_container = zeros(hidden_size, hidden_size);
        u_reset_container = zeros(input_size, hidden_size);
        b_reset_container = zeros(1 , hidden_size);
        w_update_container = zeros(hidden_size, hidden_size);
        u_update_container = zeros(input_size, hidden_size);
        b_update_container = zeros(1 , hidden_size);
        w_candidate_container = zeros(hidden_size, hidden_size);
        u_candidate_container = zeros(input_size, hidden_size);
        b_candidate_container = zeros(1 , hidden_size);
        w_output_container = zeros(hidden_size , output_size);
        b_output_container = zeros(1 , output_size);
        
        epoch_error_training = zeros(1, epoch_num);
        epoch_error_validation = zeros(1, epoch_num);
        
        % Epoch iteration
        for epoch = 1:epoch_num
            
            rnd_sample_index = randperm(2700);
            counter = 0;
            epoch_training_loss = 0;
            
            confusion_matrix_test = zeros(class_num, class_num);
            confusion_matrix_train = zeros(class_num, class_num);
            cci_training = 0;
            
            for k=1:2700
                
                sample_data = training_set(:, :, rnd_sample_index(k));
                sample_data_label = training_set_label(:, rnd_sample_index(k));
                
                % single layer forward pass
                [y, h_store, r_store, c_store, z_store] = GRU_forward(sample_data, w_reset_gate, u_reset_gate, b_reset_gate, w_output_gate, u_output_gate, b_output_gate, w_candidate_gate, u_candidate_gate, b_candidate_gate, w_y_out, b_y_out);
                
                % softmax layer
                y = y - max(y);
                o = exp(y) ./ sum(exp(y));
                
                [~, label] = max(o);
                [~, true_label] = max(sample_data_label);
                
                confusion_matrix_train(label, true_label) = confusion_matrix_train(label, true_label) + 1;
                if(label == true_label)
                    cci_training = cci_training + 1;
                end
                
                v_sgd = -(log(o)*sample_data_label);
                d_y = -sample_data_label'+o;
                epoch_training_loss = epoch_training_loss + v_sgd;
                
                % backpropagation through time GRU
                [del_w_y, del_b_y, del_w_reset_gate, del_u_reset_gate, del_b_reset_gate, del_w_output_gate, del_u_output_gate, del_b_output_gate, del_w_candidate_gate, del_u_candidate_gate, del_b_candidate_gate] = GRU_bptt(sample_data, d_y, r_store, c_store, z_store, h_store, w_reset_gate, w_output_gate, w_candidate_gate, w_y_out, hidden_size, input_size);
                counter = counter +1;
                
                w_reset_container = w_reset_container + del_w_reset_gate;
                u_reset_container =u_reset_container + del_u_reset_gate;
                b_reset_container = b_reset_container + del_b_reset_gate;
                w_update_container = w_update_container + del_w_output_gate;
                u_update_container =u_update_container + del_u_output_gate;
                b_update_container = b_update_container + del_b_output_gate;
                w_candidate_container = w_candidate_container + del_w_candidate_gate;
                u_candidate_container = u_candidate_container + del_u_candidate_gate;
                b_candidate_container = b_candidate_container + del_b_candidate_gate;
                w_output_container = w_output_container + del_w_y;
                b_output_container = b_output_container + del_b_y;
                
                if counter == batch_size
                    counter = 0;
                    
                    % updates
                    alpha_wr = alpha_wr*alpha + learning_rate * (1/batch_size) * w_reset_container;
                    alpha_ur = alpha_ur*alpha + learning_rate * (1/batch_size) * u_reset_container;
                    alpha_br = alpha_br*alpha + learning_rate * (1/batch_size) * b_reset_container;
                    alpha_wz = alpha_wz*alpha + learning_rate * (1/batch_size) * w_update_container;
                    alpha_uz = alpha_uz*alpha + learning_rate * (1/batch_size) * u_update_container;
                    alpha_bz = alpha_bz*alpha + learning_rate * (1/batch_size) * b_update_container;
                    alpha_wc = alpha_wc*alpha + learning_rate * (1/batch_size) * w_candidate_container;
                    alpha_uc = alpha_uc*alpha + learning_rate * (1/batch_size) * u_candidate_container;
                    alpha_bc = alpha_bc*alpha + learning_rate * (1/batch_size) * b_candidate_container;
                    alpha_wy = alpha_wy*alpha + learning_rate * (1/batch_size) * w_output_container;
                    alpha_by = alpha_by*alpha + learning_rate * (1/batch_size) * b_output_container;
                    
                    w_reset_gate = w_reset_gate - alpha_wr;
                    u_reset_gate = u_reset_gate - alpha_ur;
                    b_reset_gate = b_reset_gate - alpha_br;
                    w_output_gate = w_output_gate - alpha_wz;
                    u_output_gate = u_output_gate - alpha_uz;
                    b_output_gate = b_output_gate - alpha_bz;
                    w_candidate_gate = w_candidate_gate - alpha_wc;
                    u_candidate_gate = u_candidate_gate - alpha_uc;
                    b_candidate_gate = b_candidate_gate - alpha_bc;
                    w_y_out = w_y_out - alpha_wy;
                    b_y_out = b_y_out - alpha_by;
                    
                    w_reset_container = zeros(hidden_size, hidden_size);
                    u_reset_container = zeros(input_size, hidden_size);
                    b_reset_container = zeros(1 , hidden_size);
                    w_update_container = zeros(hidden_size, hidden_size);
                    u_update_container = zeros(input_size, hidden_size);
                    b_update_container = zeros(1 , hidden_size);
                    w_candidate_container = zeros(hidden_size, hidden_size);
                    u_candidate_container = zeros(input_size, hidden_size);
                    b_candidate_container = zeros(1 , hidden_size);
                    
                    w_output_container = zeros(hidden_size, output_size);
                    b_output_container = zeros(1, output_size);
                end
                
                if k == 2700
                    batch_size  = 12;
                    counter = 0;
                    
                    % updates
                    alpha_wr = alpha_wr*alpha + learning_rate * (1/batch_size) * w_reset_container;
                    alpha_ur = alpha_ur*alpha + learning_rate * (1/batch_size) * u_reset_container;
                    alpha_br = alpha_br*alpha + learning_rate * (1/batch_size) * b_reset_container;
                    alpha_wz = alpha_wz*alpha + learning_rate * (1/batch_size) * w_update_container;
                    alpha_uz = alpha_uz*alpha + learning_rate * (1/batch_size) * u_update_container;
                    alpha_bz = alpha_bz*alpha + learning_rate * (1/batch_size) * b_update_container;
                    alpha_wc = alpha_wc*alpha + learning_rate * (1/batch_size) * w_candidate_container;
                    alpha_uc = alpha_uc*alpha + learning_rate * (1/batch_size) * u_candidate_container;
                    alpha_bc = alpha_bc*alpha + learning_rate * (1/batch_size) * b_candidate_container;
                    alpha_wy = alpha_wy*alpha + learning_rate * (1/batch_size) * w_output_container;
                    alpha_by = alpha_by*alpha + learning_rate * (1/batch_size) * b_output_container;
                    
                    w_reset_gate = w_reset_gate - alpha_wr;
                    u_reset_gate = u_reset_gate - alpha_ur;
                    b_reset_gate = b_reset_gate - alpha_br;
                    w_output_gate = w_output_gate - alpha_wz;
                    u_output_gate = u_output_gate - alpha_uz;
                    b_output_gate = b_output_gate - alpha_bz;
                    w_candidate_gate = w_candidate_gate - alpha_wc;
                    u_candidate_gate = u_candidate_gate - alpha_uc;
                    b_candidate_gate = b_candidate_gate - alpha_bc;
                    w_y_out = w_y_out - alpha_wy;
                    b_y_out = b_y_out - alpha_by;
                    
                    w_reset_container = zeros(hidden_size, hidden_size);
                    u_reset_container = zeros(input_size, hidden_size);
                    b_reset_container = zeros(1 , hidden_size);
                    w_update_container = zeros(hidden_size, hidden_size);
                    u_update_container = zeros(input_size, hidden_size);
                    b_update_container = zeros(1 , hidden_size);
                    w_candidate_container = zeros(hidden_size, hidden_size);
                    u_candidate_container = zeros(input_size, hidden_size);
                    b_candidate_container = zeros(1 , hidden_size);
                    w_output_container = zeros(  hidden_size , output_size );
                    b_output_container = zeros(  1 , output_size );
                    
                    batch_size = 32;
                end
            end
            
            epoch_error_training(epoch) = epoch_training_loss/2700;
            epoch_training_loss = 0;
            
            % Validation loss and accuracy
            validation_loss = 0;
            for m = 1:300
                sample_data = validation_set(:,:,m);
                sample_data_label = validation_set_label(:,m);
                % GRU forward pass
                [y, ~, ~ , ~, ~] = GRU_forward(sample_data, w_reset_gate, u_reset_gate, b_reset_gate, w_output_gate, u_output_gate, b_output_gate, w_candidate_gate, u_candidate_gate, b_candidate_gate, w_y_out, b_y_out);
                % softmax layer
                y = y - max(y);
                o = exp(y) ./ sum(exp(y));
                v_sgd = -(log(o)*sample_data_label);
                validation_loss = validation_loss + v_sgd;
            end
            
            % Test accuracy
            cci_test = 0;
            for l = 1:600
                sample_data = tstX(:,:,l );
                sample_data_label = tstY(:,l);
                % GRU forward pass
                [y , ~ , ~ , ~, ~] = GRU_forward(sample_data, w_reset_gate, u_reset_gate, b_reset_gate, w_output_gate, u_output_gate, b_output_gate, w_candidate_gate, u_candidate_gate, b_candidate_gate, w_y_out, b_y_out);
                % oftmax layer
                y = y - max(y);
                o = exp(y) ./ sum(exp(y));
                
                [~, label] = max(o);
                [~, label_true] = max(sample_data_label);
                confusion_matrix_test(label,label_true) =  confusion_matrix_test(label,label_true) + 1;
                if(label == label_true)
                    cci_test = cci_test + 1;
                end
            end
            
            epoch_error_validation(epoch) = validation_loss/300;
            
            figure(22)
            plot(1:epoch, epoch_error_validation(1:epoch) ,'r');
            hold on
            plot( 1:epoch , epoch_error_training(1:epoch),'b');
            title("Cross Entropy vs Epoch for GRU , learning rate = " + learning_rate + ", alpha = " + alpha)
            legend("Validation Error","Training Error")
            xlabel("Epoch Number ")
            ylabel("Error")
            grid on
            
            if validation_loss/300 < 0.6
                break;
            end
        end
        
        disp("Training loss: " + epoch_error_training(epoch))
        disp("Training Accuracy (%): " + 100*cci_training/2700);
        disp("Training Confusion Matrix");
        confusion_matrix_train
        
        disp("Validation loss: " + validation_loss/300);
        disp("Test Accuracy (%) :" + 100*cci_test/600);
        disp("Test Confusion Matrix");
        confusion_matrix_test
        elapsedTime = toc/60;
        disp("Time passed in Question 3 Part C : " + elapsedTime + " min");
        
        
end
end

%% Functions

function [w_ih, w_ho, b_ih, b_ho, epoch_error] = epoch_training(params, w_ih, w_ho, b_ih, b_ho, learning_rate, batch_input, epoch_error, epoch)

We = {w_ih, w_ho, b_ih, b_ho};
[J, Jgrad] = aeCost(We, batch_input, params);

del_w_ih = cell2mat(Jgrad(1));
del_w_ho = cell2mat(Jgrad(2));
del_b_ih = cell2mat(Jgrad(3));
del_b_ho = cell2mat(Jgrad(4));

% update equations
del_w = del_w_ih + del_w_ho';
w_ih = w_ih - (1/length(batch_input)) * learning_rate * del_w;
w_ho = w_ho - (1/length(batch_input)) * learning_rate * del_w';
b_ih = b_ih - (1/length(batch_input)) * learning_rate * del_b_ih;
b_ho = b_ho - (1/length(batch_input)) * learning_rate * del_b_ho;
epoch_error(epoch) = J;
%if epoch>1
%disp( "Error Dif: " + ( error_per_epoch(epoch-1) - error_per_epoch(epoch)))
%end

end

function [J, Jgrad] = aeCost(We, batch_input, params)

w_ih = cell2mat(We(1));
w_ho = cell2mat(We(2));
b_ih = cell2mat(We(3));
b_ho = cell2mat(We(4));

% forward pass equations
v_hidden_layer = w_ih * batch_input + b_ih;
o_hidden_layer = 1./(1+exp(-v_hidden_layer));
v_output_layer = w_ho * o_hidden_layer + b_ho;
o_output_layer = 1./(1+exp(-v_output_layer));
% the average activation of hidden unit b across training samples
rho_b = (1/length(batch_input))*sum(o_hidden_layer, 2);

% cost calculation with 3 terms
mse_error_term = (0.5/length(batch_input)) * sum(sum((batch_input-o_output_layer) .* (batch_input-o_output_layer)));
tykhonov_term = (0.5*params(3)) * (sum(sum(w_ih.*w_ih)) + sum(sum(w_ho.*w_ho)));
kullback_leiber_term = params(5) * sum(params(4) * (log(params(4)) - log(rho_b)) + (1-params(4)) * (log(1-params(4))-log(1-rho_b)));

% total cost
J = mse_error_term + tykhonov_term + kullback_leiber_term;

% hidden-output layer gradient equations
sigma_ho = (-(batch_input-o_output_layer)) .* (o_output_layer.*(1-o_output_layer));
mse_gradient_ho = sigma_ho * o_hidden_layer';
tykhonov_gradient_ho = params(3) * w_ho;
KL_gradient_ho = 0;
del_J_grad_ho = mse_gradient_ho + tykhonov_gradient_ho + KL_gradient_ho;
del_J_grad_b_ho = sigma_ho * ones(length(batch_input), 1);

% input-hidden layer gradient equations
sigma_ih = w_ho' * sigma_ho .* (o_hidden_layer .* (1-o_hidden_layer));
mse_gradient_ih = sigma_ih * batch_input';
tykhonov_gradient_ih = params(3) * w_ih;
KL_gradient_ih = params(5) * ((-params(4)./rho_b)+((1-params(4))./(1-rho_b))) .* (o_hidden_layer.*(1-o_hidden_layer)) * batch_input';
del_J_grad_ih = mse_gradient_ih + tykhonov_gradient_ih + KL_gradient_ih;
del_J_grad_b_ih = sigma_ih * ones(length(batch_input), 1) + params(5) * ((-params(4)./rho_b)+((1-params(4))./(1-rho_b))) .* (o_hidden_layer.*(1-o_hidden_layer)) * ones(length(batch_input), 1);


Jgrad = {del_J_grad_ih, del_J_grad_ho, del_J_grad_b_ih, del_J_grad_b_ho};
end

%

function [o, output_store, potential] = forward_pass(sample_data, w_hh , w_ho, w_ih, bias_hidden, bias_output)

output = zeros(length(w_hh), 1);
output_store = zeros(length(w_hh), 1, 150);

for i = 1:150
    potential = w_ih * sample_data(:,i) + w_hh * output + bias_hidden;
    output = tanh(potential);
    output_store(:,:,i+1) = output;
    
end

o = w_ho * output + bias_output;

end

function [del_w_ih , del_w_hh, del_w_ho, del_bias_hidden, del_bias_output] = bptt(sample_data, w_ho, w_hh, d_y, output_store)

del_w_ho = d_y * output_store(:,:,end)';
del_bias_output = d_y;
del_w_hh = zeros(128 ,128 );
del_w_ih = zeros(128,3);
del_bias_hidden = zeros(128,1);
del_hidden = w_ho' * d_y;

time_iter = 150;

while time_iter > 90
    change = ((1 - output_store(:, :, time_iter + 1).^2) .* del_hidden);
    del_bias_hidden = del_bias_hidden + change;
    
    del_w_hh = del_w_hh + change * output_store(:,:,time_iter)';
    del_w_ih = del_w_ih + change * sample_data(:,time_iter)';
    
    del_hidden = w_hh * change;
    
    time_iter = time_iter - 1;
end

del_bias_hidden( del_bias_hidden > 1 ) = 1;
del_bias_hidden( del_bias_hidden < -1 ) = -1;
del_bias_output( del_bias_output > 1 ) = 1;
del_bias_output( del_bias_output < -1 ) = -1;
del_w_ih( del_w_ih > 1 ) = 1;
del_w_ih( del_w_ih < -1 ) = -1;
del_w_hh( del_w_hh > 1 ) = 1;
del_w_hh( del_w_hh < -1 ) = -1;
del_w_ho( del_w_ho > 1 ) = 1;
del_w_ho( del_w_ho < -1 ) = -1;

end

function [y, h_store, c_store, forget_store, input_store, output_store, g_store] = LSTM_forward(sample_data, w_forget, u_forget, b_forget, w_input, u_input, b_input, w_output, u_output, b_output, w_candidate, u_candidate, b_candidate, w_y_output, b_y_output)

h = zeros(1,128);
c = zeros(1,128);
h_store = zeros(1,128,1);
c_store = zeros(1,128,1);
forget_store = zeros(1,128,1);
input_store = zeros(1,128,1);
output_store = zeros(1,128,1);
g_store = zeros(1,128,1);

sample_data = sample_data';
for k=1:150
    f = sigmoid_function(sample_data(k,:) * u_forget + h * w_forget + b_forget);
    i = sigmoid_function(sample_data(k,:) * u_input + h * w_input + b_input);
    o = sigmoid_function(sample_data(k,:) * u_output + h * w_output + b_output);
    g = tanh(sample_data(k,:) * u_candidate + h * w_candidate + b_candidate);
    
    c_prev = c;
    c = f .* c_prev + i .* g;
    h = o .* tanh(c);
    h_store(:,:,k+1) = h;
    c_store(:,:,k+1) = c;
    forget_store(:,:,k) = f;
    input_store(:,:,k) = i;
    output_store(:,:,k) = o;
    g_store(:,:,k) = g;
end
y = h * w_y_output + b_y_output;
end

function y = sigmoid_function(x)
y = 1./(1 + exp(-x));
end

function [d_w_y, d_b_y, del_w_forget, del_u_forget, del_b_forget, del_w_input, del_u_input, del_b_input, del_w_output, del_u_output, del_b_output, del_w_gate, del_u_g, del_b_g] = LSTM_bptt(sample_data, d_y, h_store, c_store, forget_store, input_store, output_store, g_store, w_forget, w_input, w_output, w_candidate, w_y_output, hidden_size, input_size)
t = 150;
d_w_y = h_store(:,:,end)' * d_y;
d_b_y = d_y;
sample_data = sample_data';

del_w_forget = zeros(hidden_size, hidden_size);
del_u_forget = zeros(input_size, hidden_size);
del_b_forget = zeros(1, hidden_size);
del_w_input = zeros(hidden_size, hidden_size);
del_u_input = zeros(input_size, hidden_size);
del_b_input = zeros(1 , hidden_size);
del_w_output = zeros(hidden_size, hidden_size);
del_u_output = zeros(input_size, hidden_size);
del_b_output = zeros(1 , hidden_size);
del_w_gate = zeros(hidden_size, hidden_size);
del_u_g = zeros(input_size, hidden_size);
del_b_g = zeros(1 , hidden_size);
d_h = d_y * w_y_output';

while t > 0
    del_c = d_h .* (output_store(:,:,t) .* (1 - tanh(c_store(:,:,t+1))));
    del_g = d_h .* input_store(:,:,t);
    del_o = d_h .* tanh(c_store(:,:,t+1));
    del_i = del_c .* g_store(:,:,t);
    del_f = del_c .* c_store(:,:,t);
    
    del_w_input = del_w_input +  h_store(:,:,t)' * (del_i .* (1 - input_store(:,:,t)) .* input_store(:,:,t));
    del_b_input = del_b_input + del_i .* (1 - input_store(:,:,t)) .* input_store(:,:,t);
    del_u_input = del_u_input + sample_data(t,:)' * (del_i .* (1 - input_store(:,:,t)) .* input_store(:,:,t));
    del_w_forget = del_w_forget + h_store(:,:,t)' * (del_f .* (1 - forget_store(:,:,t)) .* forget_store(:,:,t) );
    del_b_forget = del_b_forget + del_f .* (1 - forget_store(:,:,t)) .* forget_store(:,:,t);
    del_u_forget = del_u_forget + sample_data(t,:)' * (del_f .* (1 - forget_store(:,:,t)) .* forget_store(:,:,t));
    del_w_output = del_w_output + h_store(:,:,t)' * (del_o .* (1 - output_store(:,:,t)) .* output_store(:,:,t));
    del_b_output = del_b_output + del_o .* (1 - output_store(:,:,t)) .* output_store(:,:,t);
    del_u_output = del_u_output + sample_data(t,:)' * (del_o .* (1 - output_store(:,:,t)) .* output_store(:,:,t));
    del_w_gate = del_w_gate + h_store(:,:,t)' * (del_g .* (1 - g_store(:,:,t).^2));
    del_b_g = del_b_g + (del_g .* (1 - g_store(:,:,t).^2));
    del_u_g = del_u_g + sample_data(t,:)' * (del_g .* (1 - g_store(:,:,t).^2));
    
    d_h = del_f .* (forget_store(:,:,t)) .* (1 - forget_store(:,:,t)) * w_forget + del_o .* (output_store(:,:,t)) .* (1 - output_store(:,:,t)) * w_output + del_i .* (input_store(:,:,t)) .* (1 - input_store(:,:,t)) * w_input + del_g .* (1 - g_store(:,:,t).^2) * w_candidate;
    
    t = t-1;
end
end

function [y, h_store, r_store, c_store, z_store] = GRU_forward(sample_data, w_reset_gate, u_reset_gate, b_reset_gate, w_output_gate, u_output_gate, b_output_gate, w_candidate_gate, u_candidate_gate, b_candidate_gate, w_y_out, b_y_out)

h_store = zeros(1,128,1);
z_store = zeros(1,128,1);
r_store = zeros(1,128,1);
c_store = zeros(1,128,1);
h = zeros(1,128);
sample_data = sample_data';

for t=1:150
    r = sigmoid_function(sample_data(t,:) * u_reset_gate + h * w_reset_gate + b_reset_gate);
    z = sigmoid_function(sample_data(t,:) * u_output_gate + h * w_output_gate + b_output_gate);
    c = tanh(sample_data(t,:) * u_candidate_gate + (r .* h) * w_candidate_gate + b_candidate_gate);
    r_store(:,:,t) = r;
    z_store(:,:,t) = z;
    c_store(:,:,t) = c;
    h = (1-z).*c + z.* h;
    h_store(:,:,t+1) = h;
end
y = h * w_y_out + b_y_out;
end

function [del_w_y, del_b_y, del_w_reset_gate, del_u_reset_gate, del_b_reset_gate, del_w_output_gate, del_u_output_gate, del_b_output_gate, del_w_candidate_gate, del_u_candidate_gate, del_b_candidate_gate] = GRU_bptt(sample_data, d_y, r_store, c_store, z_store, h_store, w_reset_gate, w_output_gate, w_container_gate, w_y_out, hidden_size, input_size)
t = 150;
del_w_y = h_store(:,:,end)' * d_y;
del_b_y = d_y;

sample_data = sample_data';

del_w_reset_gate = zeros(hidden_size, hidden_size);
del_u_reset_gate = zeros(input_size, hidden_size);
del_b_reset_gate = zeros(1 , hidden_size);

del_w_output_gate = zeros(hidden_size, hidden_size);
del_u_output_gate = zeros(input_size, hidden_size);
del_b_output_gate = zeros(1 , hidden_size);

del_w_candidate_gate = zeros(hidden_size, hidden_size);
del_u_candidate_gate = zeros(input_size, hidden_size);
del_b_candidate_gate = zeros(1, hidden_size);

d_h = d_y * w_y_out';
while t > 0
    
    d_c = d_h .* (1 - z_store(:,:,t));
    d_r = d_c .*( 1 - c_store(:,:,t).^2 ) .* r_store(:,:,t) * w_container_gate ;
    d_z = d_h .*(-c_store(:,:,t) + h_store(:,:,t));
    del_w_reset_gate = del_w_reset_gate +  h_store(:,:,t)' * (d_r .* (1 - r_store(:,:,t)) .* r_store(:,:,t));
    del_b_reset_gate = del_b_reset_gate + d_r .* (1 - r_store(:,:,t)) .* r_store(:,:,t);
    del_u_reset_gate = del_u_reset_gate + sample_data(t,:)' * (d_r .* (1 - r_store(:,:,t)) .* r_store(:,:,t));
    del_w_output_gate = del_w_output_gate + h_store(:,:,t)' * (d_z .* (1 - z_store(:,:,t)) .* z_store(:,:,t));
    del_b_output_gate = del_b_output_gate + d_z .* (1 - z_store(:,:,t)) .* z_store(:,:,t);
    del_u_output_gate = del_u_output_gate + sample_data(t,:)' * ( d_z .* (1 -  z_store(:,:,t)) .* z_store(:,:,t));
    del_w_candidate_gate = del_w_candidate_gate + ( r_store(:,:,t) .* h_store(:,:,t))' * (d_c .* (1- c_store(:,:,t).^2));
    del_b_candidate_gate = del_b_candidate_gate + d_c .* ( 1- c_store(:,:,t).^2);
    del_u_candidate_gate = del_u_candidate_gate + sample_data(t,:)' * (d_c .* (1 - c_store(:,:,t).^2));
    d_h =  d_r .* (r_store(:,:,t)) .* (1 -  r_store(:,:,t)) * w_reset_gate  + d_z .* (z_store(:,:,t)) .* (1 - z_store(:,:,t)) * w_output_gate + d_h .* (z_store(:,:,t)) + d_c .* (1 - c_store(:,:,t).^2) .* ( r_store(:,:,t)) * w_container_gate;
    
    t = t-1;
end
end

