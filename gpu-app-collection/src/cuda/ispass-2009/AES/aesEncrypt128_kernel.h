
/***************************************************************************
 *   Copyright (C) 2006                                                    *
 *                                                                         *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/


/**
	@author Svetlin Manavski <svetlin@manavski.com>
 */

/* aes encryption operation:
 * Device code.
 *
 */

#ifndef _AESENCRYPT128_KERNEL_H_
#define _AESENCRYPT128_KERNEL_H_

#include <stdio.h>

// Thread block size
#define BSIZE 256

#define STAGEBLOCK1(index)	CUT_BANK_CHECKER( stageBlock1, index )
#define STAGEBLOCK2(index)	CUT_BANK_CHECKER( stageBlock2, index )

#define TBOXE0(index)	    CUT_BANK_CHECKER( tBox0Block, index )
#define TBOXE1(index)		CUT_BANK_CHECKER( tBox1Block, index )
#define TBOXE2(index)		CUT_BANK_CHECKER( tBox2Block, index )
#define TBOXE3(index)		CUT_BANK_CHECKER( tBox3Block, index )

//texture<unsigned, 1, cudaReadModeElementType> texEKey128;

__global__ void aesEncrypt128( unsigned * result, unsigned * inData, unsigned *key, int inputSize)
{
	unsigned bx		= blockIdx.x;
    unsigned tx		= threadIdx.x;
    unsigned mod4tx = tx%4;
    unsigned int4tx = tx/4;
    unsigned idx2	= int4tx*4;
	int x;
	unsigned keyElem;
   
    __shared__ UByte4 stageBlock1[BSIZE];
	__shared__ UByte4 stageBlock2[BSIZE];


	__shared__ UByte4 tBox0Block[256];
	__shared__ UByte4 tBox1Block[256];
	__shared__ UByte4 tBox2Block[256];
	__shared__ UByte4 tBox3Block[256];

	// input caricati in memoria
	STAGEBLOCK1(tx).uival	= inData[BSIZE * bx + tx ];

	unsigned elemPerThread = 256/BSIZE;
	for (unsigned cnt=0; cnt<elemPerThread; cnt++) {
		TBOXE0(tx*elemPerThread + cnt).uival	= TBox0[tx*elemPerThread + cnt];
		TBOXE1(tx*elemPerThread + cnt).uival	= TBox1[tx*elemPerThread + cnt];
		TBOXE2(tx*elemPerThread + cnt).uival	= TBox2[tx*elemPerThread + cnt];
		TBOXE3(tx*elemPerThread + cnt).uival	= TBox3[tx*elemPerThread + cnt];
	}
	
	__syncthreads();
	
	//----------------------------------- 1st stage -----------------------------------

	x = mod4tx;
	keyElem = key[x];//tex1Dfetch(texEKey128, x);
    STAGEBLOCK2(tx).uival = STAGEBLOCK1(tx).uival ^ keyElem;

	__syncthreads();
	
	//-------------------------------- end of 1st stage --------------------------------
	

	//----------------------------------- 2nd stage -----------------------------------
	
    unsigned op1 = STAGEBLOCK2( posIdx_E[mod4tx*4]   + idx2 ).ubval[0];
	unsigned op2 = STAGEBLOCK2( posIdx_E[mod4tx*4+1] + idx2 ).ubval[1];
	unsigned op3 = STAGEBLOCK2( posIdx_E[mod4tx*4+2] + idx2 ).ubval[2];
	unsigned op4 = STAGEBLOCK2( posIdx_E[mod4tx*4+3] + idx2 ).ubval[3];
	
	op1 = TBOXE0(op1).uival;

    op2 = TBOXE1(op2).uival;

    op3 = TBOXE2(op3).uival;

    op4 = TBOXE3(op4).uival;

	x = mod4tx+4;
	keyElem = key[x];//tex1Dfetch(texEKey128, x);
	 STAGEBLOCK1(tx).uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 2nd stage --------------------------------
	
	//----------------------------------- 3th stage -----------------------------------
	
    op1 = STAGEBLOCK1( posIdx_E[mod4tx*4]   + idx2 ).ubval[0];
	op2 = STAGEBLOCK1( posIdx_E[mod4tx*4+1] + idx2 ).ubval[1];
	op3 = STAGEBLOCK1( posIdx_E[mod4tx*4+2] + idx2 ).ubval[2];
	op4 = STAGEBLOCK1( posIdx_E[mod4tx*4+3] + idx2 ).ubval[3];
	
	op1 = TBOXE0(op1).uival;

    op2 = TBOXE1(op2).uival;

    op3 = TBOXE2(op3).uival;

    op4 = TBOXE3(op4).uival;

	x = mod4tx+8;
	keyElem = key[x];//tex1Dfetch(texEKey128, x);
	 STAGEBLOCK2(tx).uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 3th stage --------------------------------

	//----------------------------------- 4th stage -----------------------------------
     
    op1 = STAGEBLOCK2( posIdx_E[mod4tx*4]   + idx2 ).ubval[0];
	op2 = STAGEBLOCK2( posIdx_E[mod4tx*4+1] + idx2 ).ubval[1];
	op3 = STAGEBLOCK2( posIdx_E[mod4tx*4+2] + idx2 ).ubval[2];
	op4 = STAGEBLOCK2( posIdx_E[mod4tx*4+3] + idx2 ).ubval[3];
	
	op1 = TBOXE0(op1).uival;

    op2 = TBOXE1(op2).uival;

    op3 = TBOXE2(op3).uival;

    op4 = TBOXE3(op4).uival;

	x = mod4tx+12;
	keyElem = key[x];//tex1Dfetch(texEKey128, x);
	 STAGEBLOCK1(tx).uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 4th stage --------------------------------

	//----------------------------------- 5th stage -----------------------------------
       
    op1 = STAGEBLOCK1( posIdx_E[mod4tx*4]   + idx2 ).ubval[0];
	op2 = STAGEBLOCK1( posIdx_E[mod4tx*4+1] + idx2 ).ubval[1];
	op3 = STAGEBLOCK1( posIdx_E[mod4tx*4+2] + idx2 ).ubval[2];
	op4 = STAGEBLOCK1( posIdx_E[mod4tx*4+3] + idx2 ).ubval[3];
	
	op1 = TBOXE0(op1).uival;

    op2 = TBOXE1(op2).uival;

    op3 = TBOXE2(op3).uival;

    op4 = TBOXE3(op4).uival;

	x = mod4tx+16;
	keyElem = key[x];//tex1Dfetch(texEKey128, x);
	 STAGEBLOCK2(tx).uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 5th stage --------------------------------
	
	//----------------------------------- 6th stage -----------------------------------
       
    op1 = STAGEBLOCK2( posIdx_E[mod4tx*4]   + idx2 ).ubval[0];
	op2 = STAGEBLOCK2( posIdx_E[mod4tx*4+1] + idx2 ).ubval[1];
	op3 = STAGEBLOCK2( posIdx_E[mod4tx*4+2] + idx2 ).ubval[2];
	op4 = STAGEBLOCK2( posIdx_E[mod4tx*4+3] + idx2 ).ubval[3];
	
	op1 = TBOXE0(op1).uival;

    op2 = TBOXE1(op2).uival;

    op3 = TBOXE2(op3).uival;

    op4 = TBOXE3(op4).uival;

	x = mod4tx+20;
	keyElem = key[x];//tex1Dfetch(texEKey128, x);
	 STAGEBLOCK1(tx).uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 6th stage --------------------------------

	//----------------------------------- 7th stage -----------------------------------
       
    op1 = STAGEBLOCK1( posIdx_E[mod4tx*4]   + idx2 ).ubval[0];
	op2 = STAGEBLOCK1( posIdx_E[mod4tx*4+1] + idx2 ).ubval[1];
	op3 = STAGEBLOCK1( posIdx_E[mod4tx*4+2] + idx2 ).ubval[2];
	op4 = STAGEBLOCK1( posIdx_E[mod4tx*4+3] + idx2 ).ubval[3];
	
	op1 = TBOXE0(op1).uival;

    op2 = TBOXE1(op2).uival;

    op3 = TBOXE2(op3).uival;

    op4 = TBOXE3(op4).uival;

	x = mod4tx+24;
	keyElem = key[x];//tex1Dfetch(texEKey128, x);
	STAGEBLOCK2(tx).uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 7th stage --------------------------------
	
	//----------------------------------- 8th stage -----------------------------------
       
    op1 = STAGEBLOCK2( posIdx_E[mod4tx*4]   + idx2 ).ubval[0];
	op2 = STAGEBLOCK2( posIdx_E[mod4tx*4+1] + idx2 ).ubval[1];
	op3 = STAGEBLOCK2( posIdx_E[mod4tx*4+2] + idx2 ).ubval[2];
	op4 = STAGEBLOCK2( posIdx_E[mod4tx*4+3] + idx2 ).ubval[3];
	
	op1 = TBOXE0(op1).uival;

    op2 = TBOXE1(op2).uival;

    op3 = TBOXE2(op3).uival;

    op4 = TBOXE3(op4).uival;

	x = mod4tx+28;
	keyElem = key[x];//tex1Dfetch(texEKey128, x);
	STAGEBLOCK1(tx).uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 8th stage --------------------------------
	
	//----------------------------------- 9th stage -----------------------------------
      
    op1 = STAGEBLOCK1( posIdx_E[mod4tx*4]   + idx2 ).ubval[0];
	op2 = STAGEBLOCK1( posIdx_E[mod4tx*4+1] + idx2 ).ubval[1];
	op3 = STAGEBLOCK1( posIdx_E[mod4tx*4+2] + idx2 ).ubval[2];
	op4 = STAGEBLOCK1( posIdx_E[mod4tx*4+3] + idx2 ).ubval[3];
	
	op1 = TBOXE0(op1).uival;

    op2 = TBOXE1(op2).uival;

    op3 = TBOXE2(op3).uival;

    op4 = TBOXE3(op4).uival;

	x = mod4tx+32;
	keyElem = key[x];//tex1Dfetch(texEKey128, x);
	STAGEBLOCK2(tx).uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 9th stage --------------------------------

	//----------------------------------- 10th stage -----------------------------------
       
    op1 = STAGEBLOCK2( posIdx_E[mod4tx*4]   + idx2 ).ubval[0];
	op2 = STAGEBLOCK2( posIdx_E[mod4tx*4+1] + idx2 ).ubval[1];
	op3 = STAGEBLOCK2( posIdx_E[mod4tx*4+2] + idx2 ).ubval[2];
	op4 = STAGEBLOCK2( posIdx_E[mod4tx*4+3] + idx2 ).ubval[3];
	
	op1 = TBOXE0(op1).uival;

    op2 = TBOXE1(op2).uival;

    op3 = TBOXE2(op3).uival;

    op4 = TBOXE3(op4).uival;

	x = mod4tx+36;
	keyElem = key[x];//tex1Dfetch(texEKey128, x);
	STAGEBLOCK1(tx).uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 10th stage --------------------------------

	//----------------------------------- 11th stage -----------------------------------
       
    op1 = STAGEBLOCK1( posIdx_E[mod4tx*4]   + idx2 ).ubval[0];
	op2 = STAGEBLOCK1( posIdx_E[mod4tx*4+1] + idx2 ).ubval[1];
	op3 = STAGEBLOCK1( posIdx_E[mod4tx*4+2] + idx2 ).ubval[2];
	op4 = STAGEBLOCK1( posIdx_E[mod4tx*4+3] + idx2 ).ubval[3];

	x = mod4tx+40;
	keyElem = key[x];//tex1Dfetch(texEKey128, x);

	
	STAGEBLOCK2(tx).ubval[3] = TBOXE1(op4).ubval[3]^( keyElem>>24);
	STAGEBLOCK2(tx).ubval[2] = TBOXE1(op3).ubval[3]^( (keyElem>>16) & 0x000000FF);
	STAGEBLOCK2(tx).ubval[1] = TBOXE1(op2).ubval[3]^( (keyElem>>8)  & 0x000000FF);
	STAGEBLOCK2(tx).ubval[0] = TBOXE1(op1).ubval[3]^( keyElem       & 0x000000FF);

	__syncthreads();

	//-------------------------------- end of 15th stage --------------------------------

	result[BSIZE * bx + tx] = STAGEBLOCK2(tx).uival;
	// end of AES
	
}

#endif // #ifndef _AESENCRYPT_KERNEL_H_
