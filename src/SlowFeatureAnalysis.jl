"""
Slow Feature Analysis
"""
module SlowFeatureAnalysis
    import MultivariateStats
    import LinearAlgebra

    """
        sfa(x)

    Each row of x is a datum.
    """
    function sfa(x)
        whitening = MultivariateStats.fit(MultivariateStats.Whitening, x, dims=1)
        whitened = MultivariateStats.transform(whitening, x)
        d_whitened = diff(whitened, dims=1)
        pca = MultivariateStats.fit(MultivariateStats.PCA, d_whitened', pratio=1)
        (pca.proj' * whitened')'
    end

    export sfa

    """
        expand_quadratic(x)

    Expand data quadratically: x -> hcat(x, x.^2, <mixed terms>). Each row of x is a datum.
    """
    function expand_quadratic(x)
        hcat(x, [x[:,i] .* x[:,j] for i in 1:size(x,2) for j in i:size(x,2)]...)
    end

    export expand_quadratic

    """
        example_data()

    Example data to play around with
    """
    function example_data()
        T = 5000;
        t = range(0, 2*π, T);

        x1 = sin.(t)+cos.(11*t).^2
        x2 = cos.(11*t)

        hcat(x1, x2);
    end

    export example_data
end
