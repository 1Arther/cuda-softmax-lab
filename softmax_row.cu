// softmax_row.cu
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void softmax_naive_kernel(const float* input,float* output,int M,int N){
    int row=blockIdx.x;
    int tid=threadIdx.x;
    if(row>=M) return;
    const float* row_input=row*N+input;
    float* row_output=row*N+output;
    extern __shared__ float sdata[];
    float max = -FLT_MAX;
    for(int i=tid;i<N;i+=blockDim.x){
        max=fmaxf(row_input[i],max);
    }
    sdata[tid]=max;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride/=2){
        if(tid<stride){
            sdata[tid]=fmaxf(sdata[tid],sdata[tid+stride]);
        }
        __syncthreads();
    }
    max=sdata[0];
    
    float sum=0.0f;
    for(int i=tid;i<N;i+=blockDim.x){
        float e=expf(row_input[i]-max);
        row_output[i]=e;
        sum+=e;
    }
    sdata[tid]=sum;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride/=2){
        if(tid<stride){
            sdata[tid]+=sdata[tid+stride];
        }
        __syncthreads();
    }   
    
    sum=sdata[0];
    for(int i=tid;i<N;i+=blockDim.x){
        row_output[i]=row_output[i]/sum;
    }
}

__inline__ __device__ float reduce_sum(float val){
    for(int offset=warpSize/2;offset>0;offset>>=1){
        val+=__shfl_down_sync(0xffffffff,val,offset);
    }
    return val;
}

__inline__ __device__ float  reduce_max(float val){
    for(int offset=warpSize/2;offset>0;offset>>=1){
        val=fmaxf(__shfl_down_sync(0xffffffff,val,offset),val);
    }
    return val;
}

__global__ void softmax_warp_kernel(const float* input,float* output,int M,int N){
    int tid=threadIdx.x;
    int row=blockIdx.x;
    if(row>=M) return;
    const float* row_input =input+row*N;
    float* row_output=output+row*N;
    int warpId=tid/warpSize;
    int laneId=tid%warpSize;
    extern __shared__ float smem[];
   
    float max=-FLT_MAX;
    for(int i=tid;i<N;i+=blockDim.x){
        max=fmaxf(max,row_input[i]);
    }
    max=reduce_max(max);
    if(laneId==0){
        smem[warpId]=max;
    }
    __syncthreads();
    if(warpId==0){
        int num=(blockDim.x+warpSize-1)/warpSize;
        max=(laneId<num)?smem[laneId]:-FLT_MAX;
        max=reduce_max(max);
    }
    if(tid==0){
        smem[0]=max;
    }
    __syncthreads();
    max=smem[0];


    float sum=0.0f;
    for(int i=tid;i<N;i+=blockDim.x){
        float e=expf(row_input[i]-max);
        row_output[i]=e;
        sum+=e;
    }
    sum=reduce_sum(sum);
    if(laneId==0){
        smem[warpId]=sum;
    }
    __syncthreads();
    if(warpId==0){
        int num=(blockDim.x+warpSize-1)/warpSize;
        sum=(laneId<num)?smem[laneId]:0.0f;
        sum=reduce_sum(sum);
    }
    if(tid==0){
        smem[0]=sum;
    }
    __syncthreads();
    sum=smem[0];
    for(int i=tid;i<N;i+=blockDim.x){
        row_output[i]=row_output[i]/sum;
    }
}

__inline__ __device__ float allreduce_sum(float val){
    val=reduce_sum(val);
    val=__shfl_sync(0xffffffff,val,0);
    return val;
}

__inline__ __device__ float allreduce_max(float val){
    val=reduce_max(val);
    val=__shfl_sync(0xffffffff,val,0);
    return val;
}

__global__ void softmax_warp_per_row_kernel(const float* input, float* output, int M, int N) {
    int tid=threadIdx.x;
    int row=tid/warpSize;
    int laneId=tid%warpSize;
    int bid=blockIdx.x;
    int warpNum=blockDim.x/warpSize;
    int global_row=bid*warpNum+row;
    if(global_row>=M) return;
    const float* row_input=input+global_row*N;
    float* row_output=output+global_row*N;
    float max=-FLT_MAX;
    for(int i=laneId;i<N;i+=warpSize){
        max=fmaxf(max,row_input[i]);
    }
    max=allreduce_max(max);
    float sum=0.0f;
    for(int i=laneId;i<N;i+=warpSize){
        float e=expf(row_input[i]-max);
        row_output[i]=e;
        sum+=e;
    }
    sum=allreduce_sum(sum);
    for(int i=laneId;i<N;i+=warpSize){
        row_output[i]/=sum;
    }
}

__global__ void softmax_vec4_kernel(const float* input, float* output, int M, int N) {
    int tid=threadIdx.x;
    int row=blockIdx.x;
    
    int N4=N/4;
    
    if(row>=M) return;
    
    int warpId=tid/warpSize;
    int laneId=tid%warpSize;
    
    extern __shared__ float smem[];
    
    const float* row_input=input+row*N;
    float* row_output=output+row*N;
    
    const float4* row_input4=reinterpret_cast<const float4*>(row_input);
    float4* row_output4=reinterpret_cast<float4*>(row_output);

    float max=-FLT_MAX;
    
    for(int i=tid;i<N4;i+=blockDim.x){
        float4 v=row_input4[i];
        max=fmaxf(max,v.x);
        max=fmaxf(max,v.y);
        max=fmaxf(max,v.z);
        max=fmaxf(max,v.w);
    }
    
    max=reduce_max(max);

    if(laneId==0){
        smem[warpId]=max;
    }
    __syncthreads();

    if(warpId==0){
        int warpNum=(blockDim.x+warpSize-1)/warpSize;
        max=laneId<warpNum?smem[laneId]:-FLT_MAX;
        max=reduce_max(max);
    }
    if(tid==0){
        smem[0]=max;
    }
    __syncthreads();
    max=smem[0];
    
    float sum=0.0f;

    for(int i=tid;i<N4;i+=blockDim.x){
        float4 v=row_input4[i];
        v.x=expf(v.x-max);
        v.y=expf(v.y-max);
        v.z=expf(v.z-max);
        v.w=expf(v.w-max);
        sum+=v.x+v.y+v.z+v.w;
        row_output4[i]=v;
    }

    sum=reduce_sum(sum);

    if(laneId==0){
        smem[warpId]=sum;
    }
    __syncthreads();

    if(warpId==0){
        int warpSum=(blockDim.x+warpSize-1)/warpSize;
        sum=laneId<warpSum?smem[laneId]:0.0f;
        sum=reduce_sum(sum);
    }
    if(tid==0){
        smem[0]=sum;
    }
    __syncthreads();
    sum=smem[0];

    for(int i=tid;i<N4;i+=blockDim.x){
        float4 v=row_output4[i];
        v.x/=sum;
        v.y/=sum;
        v.z/=sum;
        v.w/=sum;
        row_output4[i]=v;
    }
}

__global__ void softmax_vec4_fast_kernel(const float* input, float* output, int M, int N) {
    int tid=threadIdx.x;
    int row=blockIdx.x;
    
    int N4=N/4;
    
    if(row>=M) return;
    
    int warpId=tid/warpSize;
    int laneId=tid%warpSize;
    
    extern __shared__ float smem[];
    
    const float* row_input=input+row*N;
    float* row_output=output+row*N;
    
    const float4* row_input4=reinterpret_cast<const float4*>(row_input);
    float4* row_output4=reinterpret_cast<float4*>(row_output);

    float max=-FLT_MAX;
    
    for(int i=tid;i<N4;i+=blockDim.x){
        float4 v=row_input4[i];
        max=fmaxf(max,v.x);
        max=fmaxf(max,v.y);
        max=fmaxf(max,v.z);
        max=fmaxf(max,v.w);
    }
    
    max=reduce_max(max);

    if(laneId==0){
        smem[warpId]=max;
    }
    __syncthreads();

    if(warpId==0){
        int warpNum=(blockDim.x+warpSize-1)/warpSize;
        max=laneId<warpNum?smem[laneId]:-FLT_MAX;
        max=reduce_max(max);
    }
    if(tid==0){
        smem[0]=max;
    }
    __syncthreads();
    max=smem[0];
    
    float sum=0.0f;

    for(int i=tid;i<N4;i+=blockDim.x){
        float4 v=row_input4[i];
        v.x=__expf(v.x-max);
        v.y=__expf(v.y-max);
        v.z=__expf(v.z-max);
        v.w=__expf(v.w-max);
        sum+=v.x+v.y+v.z+v.w;
        row_output4[i]=v;
    }

    sum=reduce_sum(sum);

    if(laneId==0){
        smem[warpId]=sum;
    }
    __syncthreads();

    if(warpId==0){
        int warpSum=(blockDim.x+warpSize-1)/warpSize;
        sum=laneId<warpSum?smem[laneId]:0.0f;
        sum=reduce_sum(sum);
    }
    if(tid==0){
        smem[0]=sum;
    }
    __syncthreads();
    sum=smem[0];

    for(int i=tid;i<N4;i+=blockDim.x){
        float4 v=row_output4[i];
        v.x/=sum;
        v.y/=sum;
        v.z/=sum;
        v.w/=sum;
        row_output4[i]=v;
    }
}

__global__ void scaled_masked_softmax_warp_kernel(
    const float* __restrict__ input,
    const unsigned char* __restrict__ mask,
    float* __restrict__ output,
    int M,
    int N,
    float scale
){
    int tid=threadIdx.x;
    int row=blockIdx.x;
    if(row>=M) return;
    const float* row_input =input+row*N;
    float* row_output=output+row*N;
    const unsigned char* row_mask=nullptr;
    if(mask!=nullptr){
        row_mask=mask+row*N;
    }
    int warpId=tid/warpSize;
    int laneId=tid%warpSize;
    extern __shared__ float smem[];
    
    float max=-FLT_MAX;
    for(int i=tid;i<N;i+=blockDim.x){
        bool valid=(row_mask==nullptr) || (row_mask[i]!=0);
        if(valid){
            max=fmaxf(max,row_input[i]*scale);
        }
    }
    max=reduce_max(max);
    if(laneId==0){
        smem[warpId]=max;
    }
    __syncthreads();
    if(warpId==0){
        int num=(blockDim.x+warpSize-1)/warpSize;
        max=(laneId<num)?smem[laneId]:-FLT_MAX;
        max=reduce_max(max);
    }
    if(tid==0){
        smem[0]=max;
    }
    __syncthreads();
    max=smem[0];
    float sum=0.0f;
    for(int i=tid;i<N;i+=blockDim.x){
        bool valid=(row_mask==nullptr) || (row_mask[i]!=0);
        if(!valid){
            row_output[i]=0;
        }else{
            float e=expf(row_input[i]*scale-max);
            row_output[i]=e;
            sum+=e;
        }
    }
    sum=reduce_sum(sum);
    if(laneId==0){
        smem[warpId]=sum;
    }
    __syncthreads();
    if(warpId==0){
        int num=(blockDim.x+warpSize-1)/warpSize;
        sum=(laneId<num)?smem[laneId]:0.0f;
        sum=reduce_sum(sum);
    }
    if(tid==0){
        smem[0]=sum;
    }
    __syncthreads();
    sum=smem[0];
    float inv_sum=sum>0.0f?1.0f/sum:0.0f;
    for(int i=tid;i<N;i+=blockDim.x){
        row_output[i]=row_output[i]*inv_sum;
    }
}

__global__ void scaled_causal_softmax_warp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N,
    int query_len,
    float scale
){
    int tid=threadIdx.x;
    int row=blockIdx.x;
    if(row>=M) return;
    const float* row_input =input+row*N;
    float* row_output=output+row*N;
    int warpId=tid/warpSize;
    int laneId=tid%warpSize;
    extern __shared__ float smem[];
    int q=row % query_len;
    float max=-FLT_MAX;
    for(int i=tid;i<N;i+=blockDim.x){
        bool valid= i<=q;
        if(valid){
            max=fmaxf(max,row_input[i]*scale);
        }
    }
    max=reduce_max(max);
    if(laneId==0){
        smem[warpId]=max;
    }
    __syncthreads();
    if(warpId==0){
        int num=(blockDim.x+warpSize-1)/warpSize;
        max=(laneId<num)?smem[laneId]:-FLT_MAX;
        max=reduce_max(max);
    }
    if(tid==0){
        smem[0]=max;
    }
    __syncthreads();
    max=smem[0];
    float sum=0.0f;
    for(int i=tid;i<N;i+=blockDim.x){
        bool valid= i<=q;
        if(!valid){
            row_output[i]=0;
        }else{
            float e=expf(row_input[i]*scale-max);
            row_output[i]=e;
            sum+=e;
        }
    }
    sum=reduce_sum(sum);
    if(laneId==0){
        smem[warpId]=sum;
    }
    __syncthreads();
    if(warpId==0){
        int num=(blockDim.x+warpSize-1)/warpSize;
        sum=(laneId<num)?smem[laneId]:0.0f;
        sum=reduce_sum(sum);
    }
    if(tid==0){
        smem[0]=sum;
    }
    __syncthreads();
    sum=smem[0];
    float inv_sum=sum>0.0f?1.0f/sum:0.0f;
    for(int i=tid;i<N;i+=blockDim.x){
        row_output[i]=row_output[i]*inv_sum;
    }
}