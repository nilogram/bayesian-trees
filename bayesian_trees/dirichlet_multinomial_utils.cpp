//cppimport

#include <cmath>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <boost/math/special_functions/digamma.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <iomanip>

namespace py = pybind11;


double digamma(double x) {
    return boost::math::digamma(x);
}


double lgamma(double x) {
    return std::lgamma(x);
}


double dirichlet_multinomial_logl(
    const std::vector<std::vector<double>>& counts, 
    const std::vector<double>& alpha
) {
    int N = counts.size();  // number of samples
    int K = alpha.size();   // number of categories

    double res = 0.0;
    double alpha_sum = std::accumulate(alpha.begin(), alpha.end(), 0.0);
    for (int n = 0; n < N; ++n) {
        double counts_sum = std::accumulate(counts[n].begin(), counts[n].end(), 0.0);
        res += lgamma(alpha_sum) - lgamma(counts_sum + alpha_sum);
        for (int k = 0; k < K; ++k) {
            res += lgamma(counts[n][k] + alpha[k]) - lgamma(alpha[k]);
        }
    }

    return res;
}


std::vector<double> dirichlet_multinomial_grad(
    const std::vector<std::vector<double>>& counts, 
    const std::vector<double>& alpha
) {
    int N = counts.size();     // number of samples
    int K = counts[0].size();  // number of categories

    std::vector<double> res(K, 0.0);
    double alpha_sum = std::accumulate(alpha.begin(), alpha.end(), 0.0);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            double counts_sum = std::accumulate(counts[n].begin(), counts[n].end(), 0.0);
            res[k] += (
                digamma(alpha_sum)
                - digamma(counts_sum + alpha_sum)  
                + digamma(counts[n][k] + alpha[k])
                - digamma(alpha[k]) 
            );
        }
    }

    return res;
}


py::dict dirichlet_multinomial_mle(
    const std::vector<std::vector<double>>& counts, 
    const std::vector<double>& alpha_init = {}, 
    double tol = 1e-10, 
    int max_iter = 100000,
    double min_alpha = 1e-6
) {    
    int N = counts.size();              // number of samples
    int K = counts[0].size();           // number of categories
    std::vector<double> alpha(K, 1.0);  // non-informative prior

    if (!alpha_init.empty()) {
        std::copy(alpha_init.begin(), alpha_init.end(), alpha.begin());
    }

    int iter = 0;
    while (iter < max_iter) {
        std::vector<double> alpha_new(K);
        double alpha_sum = std::accumulate(alpha.begin(), alpha.end(), 0.0);
        std::vector<double> counts_alpha_sum(N);
        
        for (int k = 0; k < K; ++k) {
            double numerator = 0.0;
            double denominator = 0.0;
            for (int n = 0; n < N; ++n) {
                double counts_sum = std::accumulate(counts[n].begin(), counts[n].end(), 0.0);
                numerator += digamma(counts[n][k] + alpha[k]) - digamma(alpha[k]);
                denominator += digamma(counts_sum + alpha_sum) - digamma(alpha_sum);
            }
            alpha_new[k] = alpha[k] * (numerator / denominator);
            if (alpha_new[k] < min_alpha) {
                alpha_new[k] = min_alpha;
            }
        }
                
        double max_diff = 0.0;
        for (int k = 0; k < K; ++k) {
            max_diff = std::max(max_diff, std::abs(alpha_new[k] - alpha[k]));
        }
        
        alpha = alpha_new;
        ++iter;
        
        if (max_diff < tol) {
            break;
        }
    }

    py::dict result;    
    result["alpha"] = alpha;
    result["gradient"] = dirichlet_multinomial_grad(counts, alpha);
    result["iter"] = iter;
    
    return result;
}


PYBIND11_MODULE(dirichlet_multinomial_utils, m) {
    m.def(
        "dirichlet_multinomial_mle", 
        &dirichlet_multinomial_mle, 
        py::arg("counts"), 
        py::arg("alpha_init") = std::vector<double>{}, 
        py::arg("tol") = 1e-10, 
        py::arg("max_iter") = 100000,
        py::arg("min_alpha") = 1e-6
    );
    
    m.def(
        "dirichlet_multinomial_logl", 
        &dirichlet_multinomial_logl, 
        py::arg("counts"), 
        py::arg("alpha")
    );

    m.def(
        "dirichlet_multinomial_grad", 
        &dirichlet_multinomial_grad, 
        py::arg("counts"), 
        py::arg("alpha")
    );
}

/*
<%
setup_pybind11(cfg)
%>
*/
