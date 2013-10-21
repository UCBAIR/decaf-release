#ifndef _DECAF_NEURON_H
#define _DECAF_NEURON_H

extern "C" {

void relu_forward(const int len, const void* input, void* output, int n);

} // extern "C"

#endif // _DECAF_NEURON_H
