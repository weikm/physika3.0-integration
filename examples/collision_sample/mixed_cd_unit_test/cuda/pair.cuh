
typedef struct _g_pair {
	uint2 *_dPairs;
	uint *_dIdx;
	int _offset;
	int _length;
	int _max_length;

	void pushData(int length, void *data)
	{
		setLength(length);
		cutilSafeCall(cudaMemcpy(_dPairs, data, length * sizeof(uint2), cudaMemcpyHostToDevice));
	}

	void init(int length) {
		uint dummy[] = { 0 };

		if (_dPairs != NULL) {
			cudaFree(_dPairs);
			cudaFree(_dIdx);
		}

		cutilSafeCall(cudaMalloc((void**)&_dIdx, 1 * sizeof(uint)));
		cutilSafeCall(cudaMalloc((void**)&_dPairs, length * sizeof(uint2)));

		_length = length;
		cutilSafeCall(cudaMemcpy(_dIdx, dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemset(_dPairs, 0, length*sizeof(uint2)));
		reportMemory("g_pair.init");

		_offset = 0;
		_max_length=0;
	}

	void popData(void *data, int num){
		cutilSafeCall(cudaMemcpy(data, _dPairs, num*sizeof(uint2), cudaMemcpyDeviceToHost));
	}

	void clear() {
		uint dummy[] = { 0 };
		cutilSafeCall(cudaMemcpy(_dIdx, dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));
		_offset = 0;
	}

	void destroy() {
		cudaFree(_dPairs);
		cudaFree(_dIdx);
	}

	uint length() {
		uint dummy[] = { 0 };
		cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
		if (dummy[0] > _max_length)
			_max_length = dummy[0];

		return dummy[0];
	}

	void setLength(uint len) {
		_length = len;

		cutilSafeCall(cudaMemcpy(_dIdx, &len, 1 * sizeof(uint), cudaMemcpyHostToDevice));
	}

	int maxLength() const { return _max_length; }
} g_pair;
