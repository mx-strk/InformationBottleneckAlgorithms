#define CARD_T ${ cardinality_T }



// c_contiguous array representing a (n_rows x n_cols) matrix called array: to index (row, col) access array[row*n_cols+col]
// note: By default everything should be C contiguous. Be careful when loading matrices saved with matlab which are F contiguous
// by default !!! Double check mem flags of cl buffer and numpy array!
// Anyway: For f_contiguous arrays to index (y,x) access array[y+x*card_Y]

__kernel void compute_dkl_mat(
    const int card_T,
    const int card_Y,
    const int card_X,
    __global double* p_x_given_t,
    __global double* p_x_given_y,
    __global double* dkl_mat)
{
    int cur_x;
    int cur_y = get_global_id(0);
    int cur_t = get_global_id(1);
    double tmp;
    if ((cur_y < card_Y) && (cur_t < card_T))
    {
        tmp = 0.0;
        for (cur_x = 0; cur_x < card_X; cur_x++) {
            //double divisor = p_x_given_t[cur_t*card_X+cur_x];
            double divisor = max(1e-100,p_x_given_t[cur_t*card_X+cur_x]);

            //double argument = p_x_given_y[cur_x+cur_y*card_X] / (divisor);
            double argument = max(1e-100, p_x_given_y[cur_x+cur_y*card_X] / (divisor));

            //tmp += p_x_given_y[cur_y+cur_x*card_Y] * log2(argument);
            tmp += p_x_given_y[cur_x+cur_y*card_X] * log2(argument);

            //if ( (cur_y < 10) && (cur_t < 10) )
            //printf("card_X: %d, cur_t: %d ,cur_y: %d, cur_x: %d, p_x_given_y %f \n",card_X,cur_t,cur_y,cur_x, p_x_given_y[cur_y+cur_x*card_Y]);
            //printf("card_X: %d, cur_t: %d ,cur_y: %d, cur_x: %d, p_x_given_y %f \n",card_X,cur_t,cur_y,cur_x, p_x_given_y[cur_y*card_X+cur_x]);

        }

        //dkl_mat[cur_y+cur_t*card_Y] = tmp;
        dkl_mat[cur_y*card_T+cur_t] = tmp;
    }
}

__kernel void find_argmin(
    const int card_T,
    const int card_Y,
    __global double* dkl_mat,
    __global int* arg_min_dkl
)
{
// get row number
int i = get_global_id(0);
double tmp;
double minimum = 1000.0;
if ((i < card_Y))
{
    for (int j = 0; j < card_T; j++) {
        // read value from line
        tmp = dkl_mat[i*card_T+j];
        /*
        if (i==0){
            printf("card_T: %d, j: %d ,i: %d, dkl_mat %f \n",card_T,j,i, tmp);
        }
        */
        // check if minimum
        if (tmp < minimum)
        {
            arg_min_dkl[i] = j;
            /*
            if (i==0){
                printf("card_T: %d, j: %d ,i: %d, dkl_mat %f , argmin: %d, minimum: %f \n",card_T,j,i, tmp, arg_min_dkl[i], minimum);
            }
            */
            minimum = tmp;
        }
    }
}

}

// vv this is non sense, fixed by enforcing c contiguous arrays.
/* How to index 2D arrays
in python p(x|y) is a (card_Y,card_X) array to access (i,k) compute p_x_given_y[i+k*card_Y]
in python p(x|t) is a (card_T,card_X) array to access (j,k) compute p_x_given_t[j*card_X+k])
*/
// ^^

__kernel void allow_move(
    const int card_Y,
    __global int* argmin,
    __global int* p_t_given_y,
    __global int* length_vec)
{
    // We need to check if the new argmin would not result in an empty cluster
    //int cluster_empty[CARD_T];
    int i = get_global_id(0);
    int j;
    int desired_new_cluster;
    int old_cluster;

    if (i == 0)
    {
       for (j = 0; j < card_Y; j++) {
           //printf("i: %d, j: %d \n",i,j);
           desired_new_cluster = argmin[j];
           old_cluster = p_t_given_y[j];
           if (length_vec[old_cluster] > 1)
           {
           // the event can be moved
                length_vec[desired_new_cluster] +=1;
                length_vec[old_cluster] -=1;
                p_t_given_y[j] = desired_new_cluster;
           }
       }
    }
}

__kernel void compute_p_x_and_t_parallel(
    const int card_T,
    const int card_Y,
    const int card_X,
    __global double* p_x_and_t,
    __global double* p_x_and_y,
    __global int* ordered_cluster_location_vec,
    __global int* start_vec,
    __global int* len_vec)
{
    int cur_t = get_global_id(0);
    int cur_x = get_global_id(1);
    int cur_y;
    int number_of_events_in_cluster  = len_vec[cur_t];

    p_x_and_t[cur_t*card_X+cur_x] = 0;

    for (int cur_y_idx = 0; cur_y_idx < number_of_events_in_cluster; cur_y_idx++)
    {
       // pick element y with index start_vec_position plus current iteration
       cur_y = ordered_cluster_location_vec[start_vec[cur_t] + cur_y_idx];
       p_x_and_t[cur_t*card_X+cur_x] += p_x_and_y[cur_y*card_X+cur_x];
    }

}

__kernel void compute_p_t_parallel(
    const int card_X,
    __global double* p_x_and_t,
    __global double* p_t)
{
    int cur_t = get_global_id(0);

    p_t[cur_t] = 0;
    for (int cur_x = 0; cur_x < card_X; cur_x++){
        p_t[cur_t] += p_x_and_t[cur_t*card_X+cur_x];
    }
}


__kernel void compute_p_x_given_t_parallel(
    const int card_X,
    __global double* p_x_and_t,
    __global double* p_t)
{
    int cur_t = get_global_id(0);
    int cur_x = get_global_id(1);

    p_x_and_t[cur_t*card_X+cur_x] /= p_t[cur_t];

}


__kernel void update_distributions(
    const int card_T,
    const int card_Y,
    const int card_X,
    __global double* p_x_given_t,
    __global double* p_t,
    __global double* p_y,
    __global double* p_x_and_y,
    __global int* p_t_given_y)
{
    int k;
    int j;
    int ind;
    int i = get_global_id(0);
    if ((i < 1))
    {
        // update p(t)
        for (j = 0; j < card_Y; j++) {
            ind = p_t_given_y[j];
            p_t[ind] += p_y[j];
        }
        // update p(x|t)
        for (j = 0; j < card_Y; j++) {
            ind = p_t_given_y[j];
            for (k = 0; k < card_X; k++){
                 //p_x_given_t[ind*card_X+k] += 1/p_t[ind] * p_x_and_y[j+k*card_Y];
                 p_x_given_t[ind*card_X+k] += 1/p_t[ind] * p_x_and_y[j*card_X+k];
                 //printf("card_X: %d, j: %d ,ind: %d, k: %d, p_x_and_y %f \n",card_X,j,ind,k, p_x_and_y[j+k*card_Y]);
                 //printf("card_X: %d, j: %d ,i: %d, k: %d, p_x_given_t %f \n",card_X,j,i,k, p_x_given_t[j*card_X+k]);

            }
        }

    }
}
