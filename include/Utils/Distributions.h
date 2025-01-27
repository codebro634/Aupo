#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

namespace distr {
    double normal_quantile(double p);
    double chi2_quantile(double p, int df, bool approximate = false);
    double studt_quantile(double p, int df, bool approximate = false);
}

#endif
