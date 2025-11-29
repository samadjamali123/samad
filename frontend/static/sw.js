// Service Worker for Plant Disease Detector PWA
const CACHE_NAME = 'plant-disease-detector-v1.0.0';
const RUNTIME_CACHE = 'runtime-cache';
const IMAGE_CACHE = 'image-cache';

// Static resources to cache immediately
const STATIC_ASSETS = [
  '/',
  '/?action=new_analysis',
  '/?action=history',
  '/static/css/main.css',
  '/static/js/app.js',
  '/static/manifest.json',
  '/static/icons/icon-192x192.png',
  '/static/icons/icon-512x512.png',
  '/offline.html'
];

// Network timeout configuration
const NETWORK_TIMEOUT = 10000; // 10 seconds

// Cache size limits
const MAX_CACHE_SIZE = 50 * 1024 * 1024; // 50MB
const MAX_IMAGE_CACHE_SIZE = 20 * 1024 * 1024; // 20MB for images

// Installation event
self.addEventListener('install', (event) => {
  console.log('[SW] Installing service worker');

  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[SW] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('[SW] Static assets cached successfully');
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('[SW] Failed to cache static assets:', error);
      })
  );
});

// Activation event
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating service worker');

  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((cacheName) => cacheName !== CACHE_NAME &&
                                   cacheName !== RUNTIME_CACHE &&
                                   cacheName !== IMAGE_CACHE)
            .map((cacheName) => {
              console.log('[SW] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            })
        );
      })
      .then(() => {
        console.log('[SW] Service worker activated');
        return self.clients.claim();
      })
  );
});

// Network request interception
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests and external resources
  if (request.method !== 'GET' || !url.origin.includes(self.location.origin)) {
    return;
  }

  // Handle different request types
  if (isImageRequest(url)) {
    event.respondWith(handleImageRequest(request));
  } else if (isAPIRequest(url)) {
    event.respondWith(handleAPIRequest(request));
  } else {
    event.respondWith(handleStaticRequest(request));
  }
});

// Check if request is for an image
function isImageRequest(url) {
  return url.pathname.includes('/api/') &&
         (url.pathname.includes('analyze') || url.pathname.includes('upload'));
}

// Check if request is for API
function isAPIRequest(url) {
  return url.pathname.includes('/api/');
}

// Handle static resource requests with cache-first strategy
async function handleStaticRequest(request) {
  try {
    // Try cache first
    const cachedResponse = await caches.match(request, { cacheName: CACHE_NAME });
    if (cachedResponse) {
      console.log('[SW] Serving static from cache:', request.url);
      return cachedResponse;
    }

    // Network fallback
    console.log('[SW] Fetching static from network:', request.url);
    const networkResponse = await fetchWithTimeout(request);

    // Cache successful responses
    if (networkResponse.ok) {
      const responseClone = networkResponse.clone();
      caches.open(CACHE_NAME).then((cache) => {
        cache.put(request, responseClone);
      });
    }

    return networkResponse;
  } catch (error) {
    console.warn('[SW] Static request failed:', error);
    return getOfflineResponse(request);
  }
}

// Handle image upload/analysis requests with network-first strategy
async function handleImageRequest(request) {
  try {
    // Always try network first for image operations
    console.log('[SW] Fetching image request from network:', request.url);
    const networkResponse = await fetchWithTimeout(request);

    // Cache successful image analysis results
    if (networkResponse.ok && request.method === 'GET') {
      const responseClone = networkResponse.clone();
      caches.open(IMAGE_CACHE).then((cache) => {
        cache.put(request, responseClone).then(() => {
          // Clean old image cache if needed
          cleanImageCache();
        });
      });
    }

    return networkResponse;
  } catch (error) {
    console.warn('[SW] Image request failed, trying cache:', error);

    // Try cache for GET requests (analysis results)
    if (request.method === 'GET') {
      const cachedResponse = await caches.match(request, { cacheName: IMAGE_CACHE });
      if (cachedResponse) {
        return cachedResponse;
      }
    }

    return getOfflineResponse(request);
  }
}

// Handle API requests with network-first strategy and cache
async function handleAPIRequest(request) {
  try {
    console.log('[SW] Fetching API request from network:', request.url);
    const networkResponse = await fetchWithTimeout(request);

    // Cache successful GET requests
    if (networkResponse.ok && request.method === 'GET') {
      const responseClone = networkResponse.clone();
      caches.open(RUNTIME_CACHE).then((cache) => {
        cache.put(request, responseClone).then(() => {
          // Clean runtime cache if needed
          cleanRuntimeCache();
        });
      });
    }

    return networkResponse;
  } catch (error) {
    console.warn('[SW] API request failed, trying cache:', error);

    // Try cache for GET requests
    if (request.method === 'GET') {
      const cachedResponse = await caches.match(request, { cacheName: RUNTIME_CACHE });
      if (cachedResponse) {
        return cachedResponse;
      }
    }

    return getAPIOfflineResponse();
  }
}

// Fetch with timeout
function fetchWithTimeout(request) {
  return Promise.race([
    fetch(request),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Network timeout')), NETWORK_TIMEOUT)
    )
  ]);
}

// Get offline response for static resources
async function getOfflineResponse(request) {
  if (request.url.includes('/offline.html')) {
    return caches.match('/offline.html');
  }

  // Try to serve offline page for navigation requests
  if (request.destination === 'document') {
    const offlineResponse = await caches.match('/offline.html');
    if (offlineResponse) {
      return offlineResponse;
    }
  }

  // Return offline page or error
  return new Response('Offline - Please check your connection', {
    status: 503,
    statusText: 'Service Unavailable'
  });
}

// Get offline response for API requests
function getAPIOfflineResponse() {
  return new Response(JSON.stringify({
    success: false,
    error: 'offline',
    message: 'Currently offline. Please check your internet connection.',
    cached: true
  }), {
    status: 503,
    statusText: 'Service Unavailable',
    headers: {
      'Content-Type': 'application/json'
    }
  });
}

// Clean image cache to maintain size limits
async function cleanImageCache() {
  try {
    const cache = await caches.open(IMAGE_CACHE);
    const requests = await cache.keys();

    if (requests.length > 20) { // Keep max 20 cached images
      // Sort by timestamp if available, otherwise remove oldest
      const toDelete = requests.slice(0, requests.length - 20);
      await Promise.all(toDelete.map(req => cache.delete(req)));
    }

    // Check total cache size
    const cacheSize = await getCacheSize(IMAGE_CACHE);
    if (cacheSize > MAX_IMAGE_CACHE_SIZE) {
      // Remove oldest entries until under limit
      const requests = await cache.keys();
      for (const req of requests) {
        await cache.delete(req);
        const newSize = await getCacheSize(IMAGE_CACHE);
        if (newSize <= MAX_IMAGE_CACHE_SIZE * 0.8) break;
      }
    }
  } catch (error) {
    console.warn('[SW] Error cleaning image cache:', error);
  }
}

// Clean runtime cache to maintain size limits
async function cleanRuntimeCache() {
  try {
    const cache = await caches.open(RUNTIME_CACHE);
    const requests = await cache.keys();

    // Remove entries older than 1 hour
    const now = Date.now();
    const oneHour = 60 * 60 * 1000;

    for (const request of requests) {
      const response = await cache.match(request);
      if (response) {
        const date = response.headers.get('date');
        if (date && (now - new Date(date).getTime()) > oneHour) {
          await cache.delete(request);
        }
      }
    }

    // Check total cache size
    const cacheSize = await getCacheSize(RUNTIME_CACHE);
    if (cacheSize > MAX_CACHE_SIZE) {
      // Remove oldest entries
      const requests = await cache.keys();
      for (const req of requests) {
        await cache.delete(req);
        const newSize = await getCacheSize(RUNTIME_CACHE);
        if (newSize <= MAX_CACHE_SIZE * 0.8) break;
      }
    }
  } catch (error) {
    console.warn('[SW] Error cleaning runtime cache:', error);
  }
}

// Get cache size (approximate)
async function getCacheSize(cacheName) {
  const cache = await caches.open(cacheName);
  const requests = await cache.keys();
  let size = 0;

  for (const request of requests) {
    const response = await cache.match(request);
    if (response) {
      const responseClone = response.clone();
      const buffer = await responseClone.arrayBuffer();
      size += buffer.byteLength;
    }
  }

  return size;
}

// Handle background sync for offline actions
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync') {
    event.waitUntil(doBackgroundSync());
  }
});

// Background sync function
async function doBackgroundSync() {
  console.log('[SW] Performing background sync');

  try {
    // Get all clients and notify them
    const clients = await self.clients.matchAll();
    clients.forEach(client => {
      client.postMessage({
        type: 'BACKGROUND_SYNC',
        data: {
          message: 'Background sync completed',
          timestamp: Date.now()
        }
      });
    });
  } catch (error) {
    console.error('[SW] Background sync failed:', error);
  }
}

// Handle push notifications (future feature)
self.addEventListener('push', (event) => {
  console.log('[SW] Push message received');

  const options = {
    body: event.data ? event.data.text() : 'New analysis result available',
    icon: '/static/icons/icon-192x192.png',
    badge: '/static/icons/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'View Results',
        icon: '/static/icons/checkmark.png'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/static/icons/xmark.png'
      }
    ]
  };

  event.waitUntil(
    self.registration.showNotification('Plant Disease Detector', options)
  );
});

// Handle notification click
self.addEventListener('notificationclick', (event) => {
  console.log('[SW] Notification click received');

  event.notification.close();

  if (event.action === 'explore') {
    event.waitUntil(
      clients.openWindow('/?action=view_results')
    );
  } else if (event.action === 'close') {
    // Notification already closed
  } else {
    // Default action - open the app
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Message handling for client communication
self.addEventListener('message', (event) => {
  console.log('[SW] Message received:', event.data);

  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }

  if (event.data && event.data.type === 'GET_VERSION') {
    event.ports[0].postMessage({ version: CACHE_NAME });
  }
});

// Performance monitoring
self.addEventListener('fetch', (event) => {
  const start = performance.now();

  event.waitUntil(
    (async () => {
      try {
        const response = await fetch(event.request);
        const duration = performance.now() - start;

        // Log slow requests
        if (duration > 3000) {
          console.warn(`[SW] Slow request: ${event.request.url} took ${duration.toFixed(2)}ms`);
        }

        return response;
      } catch (error) {
        console.error(`[SW] Request failed: ${event.request.url}`, error);
        throw error;
      }
    })()
  );
});