class MinkowskiMetric:
    def __init__(self, p=2):
        '''
        Just simple init
        NOTE -> D(X,Y) = (Σ^n_n=1 |x_i - y_i|^p)^(p**-1)
        Where N is len X/Y 
        p = 1: Manhattan
        P = 2: Euclidean
        p = inf: Chebyshev
        '''
        self.p = p
        self._p_recip = 1/p if p != float('inf') else 0 #Cache reciprocal so we dont divide, absolute genius

    def calculate_distance(self, point1, point2):
        '''
            1. Check length match
            2. For loop to iterate no need for complex stuff
            3. Sum and return
        '''

        if len(point1) != len(point2):
            raise ValueError("Lists must have the same dimension")
        
        if self.p == float('inf'):
            return max(abs(a-b) for a, b in zip(point1, point2))
        elif self.p == 1:
            return sum(abs(a - b) for a, b in zip(point1, point2))
        elif self.p == 2:
            return sum((a - b) * (a - b) for a, b in zip(point1, point2)) ** self._p_recip
        
        return sum(abs(a - b) ** self.p for a, b in zip(point1, point2)) ** self._p_recip
    
    def calculate_center(self, points):
        '''
            c_j = ((1/|C|) * ∑|x_ij|^p)^(1/p)
            where |C| is number of points in cluster
        '''
        if len(points) == 0 or points.size == 0: #Thank god I added this check
            raise ValueError("Cannot calculate center of empty cluster")
            
        n_dims = len(points[0])
        n_points = len(points)
        inv_n = 1.0 / n_points
        
        #Manhattan
        if self.p == 1:
            return [sum(abs(point[j]) for point in points) * inv_n 
                   for j in range(n_dims)]
        
        #Euclidean
        if self.p == 2:
            return [sum(point[j] * point[j] for point in points) * inv_n 
                   for j in range(n_dims)]
                   
        #Chebyshev
        if self.p == float('inf'):
            return [max(abs(point[j]) for point in points) 
                   for j in range(n_dims)]
        
        #Others
        center = []
        for dim in range(n_dims): #For each dimension
            sum_power = sum(abs(point[dim]) ** self.p for point in points) #|x_ij|^p
            center.append((sum_power * inv_n) ** self._p_recip) #(1/|C| * sum)^(1/p)

        return center

'''
Too tired right now so leaving a note for future, change p so its powers of 2 so p=-2 = 2^-2 which is 0.25 which is a valid minkowski metric
'''