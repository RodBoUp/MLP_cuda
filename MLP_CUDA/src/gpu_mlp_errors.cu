#define CUDA_CHECK(call)                     \
do                                           \
{                                            \
cudaError_t err = call;                  \
\
if(err != cudaSuccess)                   \
{                                        \
std::cerr                            \
<< "CUDA Error: "                \
<< cudaGetErrorString(err)       \
<< " ("                          \
<< __FILE__                      \
<< ":"                           \
<< __LINE__                      \
<< ")"                           \
<< std::endl;                    \
\
exit(EXIT_FAILURE);                  \
}                                        \
} while(0)