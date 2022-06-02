const float loc_PI = 3.1415926535897932384626433832795f;
const float loc_SQRT1_2 = 0.70710678118654752440084436210485f;
	typedef struct {
	unsigned int workGroupShiftX;
	unsigned int workGroupShiftY;
	unsigned int workGroupShiftZ;
	}PushConsts;
	__constant__ PushConsts consts;
extern __shared__ float shared[];
extern "C" __global__ void __launch_bounds__(512) VkFFT_main_0_0 (float2* inputs, float2* outputs, float2* twiddleLUT) {
unsigned int sharedStride = 16;
float2* sdata = (float2*)shared;

	float2 temp_0;
	temp_0.x=0;
	temp_0.y=0;
	float2 temp_1;
	temp_1.x=0;
	temp_1.y=0;
	float2 temp_2;
	temp_2.x=0;
	temp_2.y=0;
	float2 temp_3;
	temp_3.x=0;
	temp_3.y=0;
	float2 temp_4;
	temp_4.x=0;
	temp_4.y=0;
	float2 temp_5;
	temp_5.x=0;
	temp_5.y=0;
	float2 temp_6;
	temp_6.x=0;
	temp_6.y=0;
	float2 temp_7;
	temp_7.x=0;
	temp_7.y=0;
	float2 w;
	w.x=0;
	w.y=0;
	float2 loc_0;
	loc_0.x=0;
	loc_0.y=0;
	float2 iw;
	iw.x=0;
	iw.y=0;
	unsigned int stageInvocationID=0;
	unsigned int blockInvocationID=0;
	unsigned int sdataID=0;
	unsigned int combinedID=0;
	unsigned int inoutID=0;
	unsigned int LUTId=0;
		if ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256) < 16384) {
		inoutID = (1 * (threadIdx.y + 0) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256));
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) + (inoutID) * 320;
			temp_0=inputs[inoutID];
		inoutID = (1 * (threadIdx.y + 32) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256));
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) + (inoutID) * 320;
			temp_1=inputs[inoutID];
		inoutID = (1 * (threadIdx.y + 64) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256));
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) + (inoutID) * 320;
			temp_2=inputs[inoutID];
		inoutID = (1 * (threadIdx.y + 96) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256));
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) + (inoutID) * 320;
			temp_3=inputs[inoutID];
		inoutID = (1 * (threadIdx.y + 128) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256));
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) + (inoutID) * 320;
			temp_4=inputs[inoutID];
		inoutID = (1 * (threadIdx.y + 160) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256));
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) + (inoutID) * 320;
			temp_5=inputs[inoutID];
		inoutID = (1 * (threadIdx.y + 192) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256));
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) + (inoutID) * 320;
			temp_6=inputs[inoutID];
		inoutID = (1 * (threadIdx.y + 224) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256));
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) + (inoutID) * 320;
			temp_7=inputs[inoutID];
	}
		if ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256) < 16384) {
		stageInvocationID = (threadIdx.y+ 0) % (1);
		LUTId = stageInvocationID + 0;
	w = twiddleLUT[LUTId];
	w.y = -w.y;
	loc_0.x = temp_4.x * w.x - temp_4.y * w.y;
	loc_0.y = temp_4.y * w.x + temp_4.x * w.y;
	temp_4.x = temp_0.x - loc_0.x;
	temp_4.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_5.x * w.x - temp_5.y * w.y;
	loc_0.y = temp_5.y * w.x + temp_5.x * w.y;
	temp_5.x = temp_1.x - loc_0.x;
	temp_5.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	loc_0.x = temp_6.x * w.x - temp_6.y * w.y;
	loc_0.y = temp_6.y * w.x + temp_6.x * w.y;
	temp_6.x = temp_2.x - loc_0.x;
	temp_6.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_3.x - loc_0.x;
	temp_7.y = temp_3.y - loc_0.y;
	temp_3.x = temp_3.x + loc_0.x;
	temp_3.y = temp_3.y + loc_0.y;
	w=twiddleLUT[LUTId+1];

	w.y = -w.y;
	loc_0.x = temp_2.x * w.x - temp_2.y * w.y;
	loc_0.y = temp_2.y * w.x + temp_2.x * w.y;
	temp_2.x = temp_0.x - loc_0.x;
	temp_2.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_3.x * w.x - temp_3.y * w.y;
	loc_0.y = temp_3.y * w.x + temp_3.x * w.y;
	temp_3.x = temp_1.x - loc_0.x;
	temp_3.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	iw.x = w.y;
	iw.y = -w.x;
	loc_0.x = temp_6.x * iw.x - temp_6.y * iw.y;
	loc_0.y = temp_6.y * iw.x + temp_6.x * iw.y;
	temp_6.x = temp_4.x - loc_0.x;
	temp_6.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	loc_0.x = temp_7.x * iw.x - temp_7.y * iw.y;
	loc_0.y = temp_7.y * iw.x + temp_7.x * iw.y;
	temp_7.x = temp_5.x - loc_0.x;
	temp_7.y = temp_5.y - loc_0.y;
	temp_5.x = temp_5.x + loc_0.x;
	temp_5.y = temp_5.y + loc_0.y;
	w=twiddleLUT[LUTId+2];

	w.y = -w.y;
	loc_0.x = temp_1.x * w.x - temp_1.y * w.y;
	loc_0.y = temp_1.y * w.x + temp_1.x * w.y;
	temp_1.x = temp_0.x - loc_0.x;
	temp_1.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	iw.x = w.y;
	iw.y = -w.x;
	loc_0.x = temp_3.x * iw.x - temp_3.y * iw.y;
	loc_0.y = temp_3.y * iw.x + temp_3.x * iw.y;
	temp_3.x = temp_2.x - loc_0.x;
	temp_3.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	iw.x = w.x * loc_SQRT1_2 + w.y * loc_SQRT1_2;
	iw.y = w.y * loc_SQRT1_2 - w.x * loc_SQRT1_2;

	loc_0.x = temp_5.x * iw.x - temp_5.y * iw.y;
	loc_0.y = temp_5.y * iw.x + temp_5.x * iw.y;
	temp_5.x = temp_4.x - loc_0.x;
	temp_5.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	w.x = iw.y;
	w.y = -iw.x;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_6.x - loc_0.x;
	temp_7.y = temp_6.y - loc_0.y;
	temp_6.x = temp_6.x + loc_0.x;
	temp_6.y = temp_6.y + loc_0.y;
	loc_0 = temp_1;
	temp_1 = temp_4;
	temp_4 = loc_0;
	loc_0 = temp_3;
	temp_3 = temp_6;
	temp_6 = loc_0;
}		sharedStride = 16;
	__syncthreads();

		if ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256) < 16384) {
	stageInvocationID = threadIdx.y + 0;
	blockInvocationID = stageInvocationID;
	stageInvocationID = stageInvocationID % 1;
	blockInvocationID = blockInvocationID - stageInvocationID;
	inoutID = blockInvocationID * 8;
	inoutID = inoutID + stageInvocationID;
	sdataID = inoutID + 0;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_0;
	sdataID = inoutID + 1;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_1;
	sdataID = inoutID + 2;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_2;
	sdataID = inoutID + 3;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_3;
	sdataID = inoutID + 4;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_4;
	sdataID = inoutID + 5;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_5;
	sdataID = inoutID + 6;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_6;
	sdataID = inoutID + 7;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_7;
}	__syncthreads();

		if ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256) < 16384) {
		stageInvocationID = (threadIdx.y+ 0) % (8);
		LUTId = stageInvocationID + 3;
		temp_0 = sdata[sharedStride*(threadIdx.y+0)+threadIdx.x];
		temp_1 = sdata[sharedStride*(threadIdx.y+32)+threadIdx.x];
		temp_2 = sdata[sharedStride*(threadIdx.y+64)+threadIdx.x];
		temp_3 = sdata[sharedStride*(threadIdx.y+96)+threadIdx.x];
		temp_4 = sdata[sharedStride*(threadIdx.y+128)+threadIdx.x];
		temp_5 = sdata[sharedStride*(threadIdx.y+160)+threadIdx.x];
		temp_6 = sdata[sharedStride*(threadIdx.y+192)+threadIdx.x];
		temp_7 = sdata[sharedStride*(threadIdx.y+224)+threadIdx.x];
	w = twiddleLUT[LUTId];
	w.y = -w.y;
	loc_0.x = temp_4.x * w.x - temp_4.y * w.y;
	loc_0.y = temp_4.y * w.x + temp_4.x * w.y;
	temp_4.x = temp_0.x - loc_0.x;
	temp_4.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_5.x * w.x - temp_5.y * w.y;
	loc_0.y = temp_5.y * w.x + temp_5.x * w.y;
	temp_5.x = temp_1.x - loc_0.x;
	temp_5.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	loc_0.x = temp_6.x * w.x - temp_6.y * w.y;
	loc_0.y = temp_6.y * w.x + temp_6.x * w.y;
	temp_6.x = temp_2.x - loc_0.x;
	temp_6.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_3.x - loc_0.x;
	temp_7.y = temp_3.y - loc_0.y;
	temp_3.x = temp_3.x + loc_0.x;
	temp_3.y = temp_3.y + loc_0.y;
	w=twiddleLUT[LUTId+8];

	w.y = -w.y;
	loc_0.x = temp_2.x * w.x - temp_2.y * w.y;
	loc_0.y = temp_2.y * w.x + temp_2.x * w.y;
	temp_2.x = temp_0.x - loc_0.x;
	temp_2.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_3.x * w.x - temp_3.y * w.y;
	loc_0.y = temp_3.y * w.x + temp_3.x * w.y;
	temp_3.x = temp_1.x - loc_0.x;
	temp_3.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	iw.x = w.y;
	iw.y = -w.x;
	loc_0.x = temp_6.x * iw.x - temp_6.y * iw.y;
	loc_0.y = temp_6.y * iw.x + temp_6.x * iw.y;
	temp_6.x = temp_4.x - loc_0.x;
	temp_6.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	loc_0.x = temp_7.x * iw.x - temp_7.y * iw.y;
	loc_0.y = temp_7.y * iw.x + temp_7.x * iw.y;
	temp_7.x = temp_5.x - loc_0.x;
	temp_7.y = temp_5.y - loc_0.y;
	temp_5.x = temp_5.x + loc_0.x;
	temp_5.y = temp_5.y + loc_0.y;
	w=twiddleLUT[LUTId+16];

	w.y = -w.y;
	loc_0.x = temp_1.x * w.x - temp_1.y * w.y;
	loc_0.y = temp_1.y * w.x + temp_1.x * w.y;
	temp_1.x = temp_0.x - loc_0.x;
	temp_1.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	iw.x = w.y;
	iw.y = -w.x;
	loc_0.x = temp_3.x * iw.x - temp_3.y * iw.y;
	loc_0.y = temp_3.y * iw.x + temp_3.x * iw.y;
	temp_3.x = temp_2.x - loc_0.x;
	temp_3.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	iw.x = w.x * loc_SQRT1_2 + w.y * loc_SQRT1_2;
	iw.y = w.y * loc_SQRT1_2 - w.x * loc_SQRT1_2;

	loc_0.x = temp_5.x * iw.x - temp_5.y * iw.y;
	loc_0.y = temp_5.y * iw.x + temp_5.x * iw.y;
	temp_5.x = temp_4.x - loc_0.x;
	temp_5.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	w.x = iw.y;
	w.y = -iw.x;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_6.x - loc_0.x;
	temp_7.y = temp_6.y - loc_0.y;
	temp_6.x = temp_6.x + loc_0.x;
	temp_6.y = temp_6.y + loc_0.y;
	loc_0 = temp_1;
	temp_1 = temp_4;
	temp_4 = loc_0;
	loc_0 = temp_3;
	temp_3 = temp_6;
	temp_6 = loc_0;
}	__syncthreads();

		if ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256) < 16384) {
	stageInvocationID = threadIdx.y + 0;
	blockInvocationID = stageInvocationID;
	stageInvocationID = stageInvocationID % 8;
	blockInvocationID = blockInvocationID - stageInvocationID;
	inoutID = blockInvocationID * 8;
	inoutID = inoutID + stageInvocationID;
	sdataID = inoutID + 0;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_0;
	sdataID = inoutID + 8;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_1;
	sdataID = inoutID + 16;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_2;
	sdataID = inoutID + 24;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_3;
	sdataID = inoutID + 32;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_4;
	sdataID = inoutID + 40;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_5;
	sdataID = inoutID + 48;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_6;
	sdataID = inoutID + 56;
	sdataID = sharedStride * sdataID;
	sdataID = sdataID + threadIdx.x;
	sdata[sdataID] = temp_7;
}	__syncthreads();

		if ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256) < 16384) {
		stageInvocationID = (threadIdx.y+ 0) % (64);
		LUTId = stageInvocationID + 27;
		temp_0 = sdata[sharedStride*(threadIdx.y+0)+threadIdx.x];
		temp_2 = sdata[sharedStride*(threadIdx.y+64)+threadIdx.x];
		temp_4 = sdata[sharedStride*(threadIdx.y+128)+threadIdx.x];
		temp_6 = sdata[sharedStride*(threadIdx.y+192)+threadIdx.x];
	w = twiddleLUT[LUTId];
	w.y = -w.y;
	loc_0.x = temp_4.x * w.x - temp_4.y * w.y;
	loc_0.y = temp_4.y * w.x + temp_4.x * w.y;
	temp_4.x = temp_0.x - loc_0.x;
	temp_4.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = temp_6.x * w.x - temp_6.y * w.y;
	loc_0.y = temp_6.y * w.x + temp_6.x * w.y;
	temp_6.x = temp_2.x - loc_0.x;
	temp_6.y = temp_2.y - loc_0.y;
	temp_2.x = temp_2.x + loc_0.x;
	temp_2.y = temp_2.y + loc_0.y;
	w=twiddleLUT[LUTId+64];
	w.y = -w.y;
	loc_0.x = temp_2.x * w.x - temp_2.y * w.y;
	loc_0.y = temp_2.y * w.x + temp_2.x * w.y;
	temp_2.x = temp_0.x - loc_0.x;
	temp_2.y = temp_0.y - loc_0.y;
	temp_0.x = temp_0.x + loc_0.x;
	temp_0.y = temp_0.y + loc_0.y;
	loc_0.x = w.x;	w.x = w.y;
	w.y = -loc_0.x;
	loc_0.x = temp_6.x * w.x - temp_6.y * w.y;
	loc_0.y = temp_6.y * w.x + temp_6.x * w.y;
	temp_6.x = temp_4.x - loc_0.x;
	temp_6.y = temp_4.y - loc_0.y;
	temp_4.x = temp_4.x + loc_0.x;
	temp_4.y = temp_4.y + loc_0.y;
	loc_0 = temp_2;
	temp_2 = temp_4;
	temp_4 = loc_0;
		stageInvocationID = (threadIdx.y+ 32) % (64);
		LUTId = stageInvocationID + 27;
		temp_1 = sdata[sharedStride*(threadIdx.y+32)+threadIdx.x];
		temp_3 = sdata[sharedStride*(threadIdx.y+96)+threadIdx.x];
		temp_5 = sdata[sharedStride*(threadIdx.y+160)+threadIdx.x];
		temp_7 = sdata[sharedStride*(threadIdx.y+224)+threadIdx.x];
	w = twiddleLUT[LUTId];
	w.y = -w.y;
	loc_0.x = temp_5.x * w.x - temp_5.y * w.y;
	loc_0.y = temp_5.y * w.x + temp_5.x * w.y;
	temp_5.x = temp_1.x - loc_0.x;
	temp_5.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_3.x - loc_0.x;
	temp_7.y = temp_3.y - loc_0.y;
	temp_3.x = temp_3.x + loc_0.x;
	temp_3.y = temp_3.y + loc_0.y;
	w=twiddleLUT[LUTId+64];
	w.y = -w.y;
	loc_0.x = temp_3.x * w.x - temp_3.y * w.y;
	loc_0.y = temp_3.y * w.x + temp_3.x * w.y;
	temp_3.x = temp_1.x - loc_0.x;
	temp_3.y = temp_1.y - loc_0.y;
	temp_1.x = temp_1.x + loc_0.x;
	temp_1.y = temp_1.y + loc_0.y;
	loc_0.x = w.x;	w.x = w.y;
	w.y = -loc_0.x;
	loc_0.x = temp_7.x * w.x - temp_7.y * w.y;
	loc_0.y = temp_7.y * w.x + temp_7.x * w.y;
	temp_7.x = temp_5.x - loc_0.x;
	temp_7.y = temp_5.y - loc_0.y;
	temp_5.x = temp_5.x + loc_0.x;
	temp_5.y = temp_5.y + loc_0.y;
	loc_0 = temp_3;
	temp_3 = temp_5;
	temp_5 = loc_0;
}		sharedStride = 16;
		if ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256) < 16384) {
}		if ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)+(((threadIdx.x + blockIdx.x * blockDim.x)) / 320) * (256) < 16384) {
		inoutID = (threadIdx.y + 0) * (64) + ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)) * (64) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320);
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) * 1 + (inoutID) * 320;
			outputs[inoutID] = temp_0;
		inoutID = (threadIdx.y + 32) * (64) + ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)) * (64) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320);
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) * 1 + (inoutID) * 320;
			outputs[inoutID] = temp_1;
		inoutID = (threadIdx.y + 64) * (64) + ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)) * (64) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320);
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) * 1 + (inoutID) * 320;
			outputs[inoutID] = temp_2;
		inoutID = (threadIdx.y + 96) * (64) + ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)) * (64) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320);
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) * 1 + (inoutID) * 320;
			outputs[inoutID] = temp_3;
		inoutID = (threadIdx.y + 128) * (64) + ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)) * (64) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320);
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) * 1 + (inoutID) * 320;
			outputs[inoutID] = temp_4;
		inoutID = (threadIdx.y + 160) * (64) + ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)) * (64) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320);
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) * 1 + (inoutID) * 320;
			outputs[inoutID] = temp_5;
		inoutID = (threadIdx.y + 192) * (64) + ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)) * (64) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320);
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) * 1 + (inoutID) * 320;
			outputs[inoutID] = temp_6;
		inoutID = (threadIdx.y + 224) * (64) + ((((threadIdx.x + blockIdx.x * blockDim.x)) / 320) % (1)) * (64) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 320);
			inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (320)) * 1 + (inoutID) * 320;
			outputs[inoutID] = temp_7;
	}
}
