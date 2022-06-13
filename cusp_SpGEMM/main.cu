#include <cusp/csr_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/gallery/poisson.h>
#include <cusp/multiply.h>
#include <cusp/print.h>

using namespace std;

typedef int     index_type;
typedef float   value_type;

typedef cusp::csr_matrix<index_type,value_type,cusp::host_memory>   CSRHost;
typedef cusp::coo_matrix<index_type,value_type,cusp::host_memory>   COOHost;

int benchmark_spgemm(char *dataset_name1,
                     char *dataset_name2)
{
    CSRHost A;
    CSRHost B;
    CSRHost C;

    if (strcmp(dataset_name1, "1") == 0)
    {
        cusp::gallery::poisson5pt(A, 256, 256);
        cusp::gallery::poisson5pt(B, 256, 256);
        cout << "2D FD, 5-point. ";
    }
    else if (strcmp(dataset_name1, "2") == 0)
    {
        cusp::gallery::poisson9pt(A, 256, 256);
        cusp::gallery::poisson9pt(B, 256, 256);
        cout << "2D FE, 9-point. ";
    }
    else if (strcmp(dataset_name1, "3") == 0)
    {
        cusp::gallery::poisson7pt(A, 51, 51, 51);
        cusp::gallery::poisson7pt(B, 51, 51, 51);
        cout << "3D FD, 7-point. ";
    }
    else if (strcmp(dataset_name1, "4") == 0)
    {
        cusp::gallery::poisson27pt(A, 51, 51, 51);
        cusp::gallery::poisson27pt(B, 51, 51, 51);
        cout << "3D FE, 27-point. ";
    }
    else
    {
        cout << " A: " << dataset_name1 << endl;
        COOHost ADup;
        cusp::io::read_matrix_market_file(ADup, dataset_name1);
		cusp::convert(ADup, A);
		
        COOHost BDup;
        cout << " B: " << dataset_name2 << endl;
        cusp::io::read_matrix_market_file(BDup, dataset_name2);
		cusp::convert(BDup, B);
/*
        ref_spgemm *ref_comp = new ref_spgemm();
        ref_comp->csr_sort_indices<index_type, value_type>(A.num_rows, &A.row_offsets[0], &A.column_indices[0], &A.values[0]);
        ref_comp->csr_sort_indices<index_type, value_type>(B.num_rows, &B.row_offsets[0], &B.column_indices[0], &B.values[0]);
*/
        //B = A;
    }

    int m = A.num_rows;
    int k = A.num_cols;
    int n = B.num_cols;

    // A
    int nnzA = A.num_entries;

    // B
    int nnzB = B.num_entries;

    cout << " A: ( " << m << " by " << k << ", nnz = " << nnzA << " ) " << endl;
    cout << " B: ( " << k << " by " << n << ", nnz = " << nnzB << " ) " << endl;

    // Multiply
	cusp::multiply(A, B, C);
	
	
    //cusp::print(C);

    return 0;
}
int main(int argc, char ** argv)
{
    // read arguments
    char *dataset_name1;
    char *dataset_name2;

	if(argc < 2) {
		printf("Format: %s dataset1 [dataset2]\n", argv[0]);
		return 0;
	}
	
    dataset_name1 = argv[1];

    if(argc > 2)
        dataset_name2 = argv[2];
    else
        dataset_name2 = dataset_name1;

    // launch testing dataset or benchmark datasets
    cout << "------------------------" << endl;
    int err = 0;
    err = benchmark_spgemm(dataset_name1, dataset_name2);

    if (err != 0) cout << "Found an err, code = " << err << endl;
    cout << "------------------------" << endl;

    return 0;
}
