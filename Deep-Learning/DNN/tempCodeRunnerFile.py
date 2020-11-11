AL, Y_assess, caches = L_model_backward_test_case()
grads = model.backward_propagation(AL, Y_assess, caches)
print_grads(grads)