// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::Result;
use cudarc::driver::{CudaDevice, CudaStream};
use parking_lot::Mutex;
use std::sync::Arc;

/// CUDA stream pool for managing async execution
pub struct CudaStreamPool {
    device: Arc<CudaDevice>,
    available_streams: Arc<Mutex<Vec<CudaStream>>>,
    max_streams: usize,
}

impl CudaStreamPool {
    /// Create a new CUDA stream pool
    #[inline(always)]
    pub fn new(device: Arc<CudaDevice>, max_streams: usize) -> Self {
        Self {
            device,
            available_streams: Arc::new(Mutex::new(Vec::new())),
            max_streams,
        }
    }

    /// Get a stream from the pool or create a new one
    #[inline(always)]
    pub fn get_stream(&self) -> Result<CudaStream> {
        let mut streams = self.available_streams.lock();

        if let Some(stream) = streams.pop() {
            Ok(stream)
        } else {
            // Create a new stream if we haven't reached the limit
            self.device.fork_default_stream().map_err(|e| {
                crate::error::MinitensorError::backend_error(
                    "CUDA",
                    format!("Failed to create CUDA stream: {}", e),
                )
            })
        }
    }

    /// Return a stream to the pool
    #[inline(always)]
    pub fn return_stream(&self, stream: CudaStream) {
        let mut streams = self.available_streams.lock();

        // Only keep streams if we haven't exceeded the max
        if streams.len() < self.max_streams {
            streams.push(stream);
        }
        // Otherwise, let the stream drop and be destroyed
    }

    /// Get the number of available streams in the pool
    #[inline(always)]
    pub fn available_count(&self) -> usize {
        self.available_streams.lock().len()
    }

    /// Synchronize all streams in the pool
    #[inline(always)]
    pub fn synchronize_all(&self) -> Result<()> {
        let streams = self.available_streams.lock();

        for stream in streams.iter() {
            stream.synchronize().map_err(|e| {
                crate::error::MinitensorError::backend_error(
                    "CUDA",
                    format!("Stream synchronization failed: {}", e),
                )
            })?;
        }

        Ok(())
    }
}

/// RAII wrapper for CUDA streams that automatically returns to pool
pub struct PooledCudaStream {
    stream: Option<CudaStream>,
    pool: Arc<CudaStreamPool>,
}

impl PooledCudaStream {
    /// Create a new pooled stream
    #[inline(always)]
    pub fn new(pool: Arc<CudaStreamPool>) -> Result<Self> {
        let stream = pool.get_stream()?;
        Ok(Self {
            stream: Some(stream),
            pool,
        })
    }

    /// Get the underlying CUDA stream
    #[inline(always)]
    pub fn stream(&self) -> &CudaStream {
        self.stream.as_ref().unwrap()
    }

    /// Take the stream out of the wrapper (prevents automatic return to pool)
    #[inline(always)]
    pub fn take(mut self) -> CudaStream {
        self.stream.take().unwrap()
    }
}

impl Drop for PooledCudaStream {
    fn drop(&mut self) {
        if let Some(stream) = self.stream.take() {
            self.pool.return_stream(stream);
        }
    }
}

/// Async execution context for CUDA operations
pub struct CudaExecutionContext {
    stream_pool: Arc<CudaStreamPool>,
    current_stream: Option<PooledCudaStream>,
}

impl CudaExecutionContext {
    /// Create a new execution context
    #[inline(always)]
    pub fn new(device: Arc<CudaDevice>, max_streams: usize) -> Self {
        let stream_pool = Arc::new(CudaStreamPool::new(device, max_streams));

        Self {
            stream_pool,
            current_stream: None,
        }
    }

    /// Get or create a stream for the current context
    #[inline(always)]
    pub fn get_stream(&mut self) -> Result<&CudaStream> {
        if self.current_stream.is_none() {
            self.current_stream = Some(PooledCudaStream::new(self.stream_pool.clone())?);
        }

        Ok(self.current_stream.as_ref().unwrap().stream())
    }

    /// Synchronize the current stream
    #[inline(always)]
    pub fn synchronize(&self) -> Result<()> {
        if let Some(ref stream) = self.current_stream {
            stream.stream().synchronize().map_err(|e| {
                crate::error::MinitensorError::backend_error(
                    "CUDA",
                    format!("Stream synchronization failed: {}", e),
                )
            })?;
        }
        Ok(())
    }

    /// Release the current stream back to the pool
    #[inline(always)]
    pub fn release_stream(&mut self) {
        self.current_stream = None;
    }

    /// Get the stream pool
    #[inline(always)]
    pub fn stream_pool(&self) -> &Arc<CudaStreamPool> {
        &self.stream_pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;

    #[test]
    fn test_stream_pool() {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let pool = CudaStreamPool::new(device, 4);

            // Test getting and returning streams
            let stream1 = pool.get_stream().unwrap();
            let stream2 = pool.get_stream().unwrap();

            pool.return_stream(stream1);
            pool.return_stream(stream2);

            assert_eq!(pool.available_count(), 2);
        }
    }

    #[test]
    fn test_pooled_stream() {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let pool = Arc::new(CudaStreamPool::new(device, 4));

            {
                let _pooled_stream = PooledCudaStream::new(pool.clone()).unwrap();
                // Stream should be automatically returned when dropped
            }

            assert_eq!(pool.available_count(), 1);
        }
    }

    #[test]
    fn test_execution_context() {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let mut context = CudaExecutionContext::new(device, 4);

            // Test getting a stream
            let _stream = context.get_stream().unwrap();

            // Test synchronization
            context.synchronize().unwrap();

            // Test releasing stream
            context.release_stream();
        }
    }
}
