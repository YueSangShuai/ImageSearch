// 全局变量
let socket;
let imageCache = {};
let pendingImages = new Set();

// 初始化 Socket.IO 连接
function initSocketConnection() {
    socket = io();
    
    // 监听图片准备完成事件
    socket.on('image_ready', function(data) {
        console.log('图片准备完成:', data);
        
        // 更新图片缓存
        imageCache[data.cache_key] = data.path;
        
        // 更新所有使用此图片的元素
        updateImageElements(data.cache_key, data.path);
    });
    
    socket.on('connect', function() {
        console.log('已连接到服务器');
    });
    
    socket.on('disconnect', function() {
        console.log('与服务器断开连接');
    });
}

// 更新使用特定缓存键的所有图片元素
function updateImageElements(cacheKey, imagePath) {
    console.log('更新图片元素:', cacheKey, imagePath);
    
    // 查找所有等待此图片的元素
    const elements = document.querySelectorAll(`img[data-cache-key="${cacheKey}"]`);
    console.log(`找到 ${elements.length} 个等待更新的元素`);
    
    if (elements.length === 0) {
        // 尝试使用文件名作为缓存键查找
        const filename = getFilenameFromPath(cacheKey);
        const filenameElements = document.querySelectorAll(`img[data-cache-key="${filename}"]`);
        console.log(`使用文件名 ${filename} 找到 ${filenameElements.length} 个元素`);
        
        if (filenameElements.length > 0) {
            updateElements(filenameElements, imagePath);
        }
    } else {
        updateElements(elements, imagePath);
    }
    
    // 检查模态框中的图片是否需要更新
    const modalImage = document.getElementById('modal-image');
    if (modalImage && modalImage.dataset.cacheKey === cacheKey) {
        console.log('更新模态框图片');
        updateModalImage(imagePath);
    }
}

// 辅助函数：更新元素集合中的所有图片
function updateElements(elements, imagePath) {
    elements.forEach(img => {
        // 创建新图片对象预加载
        const newImg = new Image();
        
        // 确保路径格式正确
        let fullPath;
        if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) {
            fullPath = imagePath;
        } else if (imagePath.startsWith('/')) {
            fullPath = imagePath;
        } else {
            fullPath = `/images/${imagePath}`;
        }
        
        newImg.onload = function() {
            // 图片加载完成后，更新原始元素
            console.log('图片加载完成:', fullPath);
            img.src = fullPath;
            img.classList.add('loaded');
            
            // 从待处理集合中移除
            pendingImages.delete(img.dataset.cacheKey);
        };
        
        newImg.onerror = function() {
            console.error('图片加载失败:', fullPath);
            // 加载失败时使用默认图片
            img.src = '/static/img/no-image.png';
        };
        
        // 加载新图片
        newImg.src = fullPath;
    });
}

// 更新模态框中的图片
function updateModalImage(imagePath) {
    const modalImage = document.getElementById('modal-image');
    if (!modalImage) return;
    
    // 确保路径格式正确
    let fullPath;
    if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) {
        fullPath = imagePath;
    } else if (imagePath.startsWith('/')) {
        fullPath = imagePath;
    } else {
        fullPath = `/images/${imagePath}`;
    }
    
    modalImage.src = fullPath;
    modalImage.classList.remove('loading');
    modalImage.classList.add('loaded');
}

// 在文档加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const queryInput = document.getElementById('query-input');
    const searchButton = document.getElementById('search-button');
    const clearButton = document.getElementById('clear-button');
    const topKInput = document.getElementById('top-k');
    const thresholdInput = document.getElementById('threshold');
    const recordCountDisplay = document.getElementById('record-count');
    const galleryContainer = document.getElementById('gallery-container');
    const loadingIndicator = document.getElementById('loading-indicator');
    const shortcutButtons = document.querySelectorAll('.shortcut-btn');
    const collectionCheckboxes = document.getElementById('collection-checkboxes');
    
    // Image search elements
    const imageUpload = document.getElementById('image-upload');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const imageSearchButton = document.getElementById('image-search-button');
    const imageClearButton = document.getElementById('image-clear-button');
    const imageTopKInput = document.getElementById('image-top-k');
    const imageThresholdInput = document.getElementById('image-threshold');
    const personThresholdInput = document.getElementById('person-threshold');
    const imageGalleryContainer = document.getElementById('image-gallery-container');
    const imageLoadingIndicator = document.getElementById('image-loading-indicator');
    const sampleImages = document.querySelectorAll('.sample-image');
    const imageCollectionCheckboxes = document.getElementById('image-collection-checkboxes');
    
    // Modal elements
    const imageModal = new bootstrap.Modal(document.getElementById('imageModal'));
    const modalImage = document.getElementById('modal-image');
    const modalCaption = document.getElementById('modal-caption');
    const modalTitle = document.getElementById('imageModalLabel');
    
    // Event listeners for text search
    searchButton.addEventListener('click', performSearch);
    queryInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
    clearButton.addEventListener('click', clearTextSearch);
    
    // Event listeners for shortcuts
    shortcutButtons.forEach(button => {
        button.addEventListener('click', function() {
            queryInput.value = this.dataset.query;
            performSearch();
        });
    });
    
    // Event listeners for image search
    imageUpload.addEventListener('change', handleImageUpload);
    imageSearchButton.addEventListener('click', performImageSearch);
    imageClearButton.addEventListener('click', clearImageSearch);
    
    // Event listeners for sample images
    sampleImages.forEach(img => {
        img.addEventListener('click', function() {
            // 确保路径格式正确
            let imagePath = this.dataset.path;
            // 如果路径不是以/images/开头，则添加前缀
            if (!imagePath.startsWith('/images/') && !imagePath.startsWith('http')) {
                // 如果是相对路径，转换为/images/路径
                imagePath = '/images/' + imagePath.replace(/^data\//, '');
            }
            loadSampleImage(imagePath);
        });
        
        // 修复样例图片的src属性
        if (!img.src.startsWith('/images/') && !img.src.startsWith('http')) {
            // 如果是相对路径，转换为/images/路径
            img.src = '/images/' + img.src.replace(/^data\//, '');
        }
    });
    
    // Add admin button to header
    const header = document.querySelector('header');
    const adminBtn = document.createElement('button');
    adminBtn.className = 'btn btn-sm btn-outline-secondary position-absolute top-0 end-0 mt-2 me-2';
    adminBtn.textContent = '管理';
    adminBtn.setAttribute('data-bs-toggle', 'modal');
    adminBtn.setAttribute('data-bs-target', '#adminModal');
    header.appendChild(adminBtn);
    
    // Initialize Socket.IO connection
    initSocketConnection();
    
    // 初始化集合复选框
    initCollections();
    
    // 监听集合切换事件
    collectionCheckboxes.addEventListener('change', function(e) {
        if (e.target.classList.contains('collection-checkbox')) {
            switchCollection(Array.from(collectionCheckboxes.querySelectorAll('input:checked')).map(checkbox => checkbox.value));
        }
    });
    
    // 图片搜索的集合切换
    imageCollectionCheckboxes.addEventListener('change', function(e) {
        if (e.target.classList.contains('image-collection-checkbox')) {
            switchCollection(Array.from(imageCollectionCheckboxes.querySelectorAll('input:checked')).map(checkbox => checkbox.value));
        }
    });
    
    // Function to perform text search
    function performSearch() {
        const query = queryInput.value.trim();
        if (!query) {
            alert('请输入搜索词');
            return;
        }
        
        // 获取选中的集合
        const selectedCollections = Array.from(collectionCheckboxes.querySelectorAll('input:checked')).map(checkbox => checkbox.value);
        
        // Show loading indicator
        galleryContainer.innerHTML = '';
        loadingIndicator.classList.remove('d-none');
        
        // Get search parameters
        const topK = parseInt(topKInput.value);
        const threshold = parseFloat(thresholdInput.value);
        
        // Perform search
        fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                top_k: topK,
                threshold: threshold,
                collections: selectedCollections
            })
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            loadingIndicator.classList.add('d-none');
            
            // Display results
            displayTextSearchResults(data);
        })
        .catch(error => {
            console.error('搜索出错:', error);
            loadingIndicator.classList.add('d-none');
            galleryContainer.innerHTML = `<div class="alert alert-danger">搜索出错: ${error}</div>`;
        });
    }
    
    // Function to display text search results
    function displayTextSearchResults(data) {
        galleryContainer.innerHTML = '';
        
        const results = data.results;
        if (results.length === 0) {
            galleryContainer.innerHTML = '<p class="no-results">没有找到匹配的结果</p>';
            return;
        }
        
        // 创建结果容器
        const resultsContainer = document.createElement('div');
        resultsContainer.className = 'results-container';
        
        // 添加结果信息
        const infoText = document.createElement('div');
        infoText.className = 'results-info';
        infoText.innerHTML = `<div class="alert alert-info">
            找到 ${results.length} 个结果，
            搜索库图片数量: ${data.record_count || 0}，
            搜索时间: ${data.search_time.toFixed(3)}秒，
            总时间: ${data.total_time.toFixed(3)}秒
        </div>`;
        
        galleryContainer.appendChild(infoText);
        
        // 创建图片网格
        const galleryGrid = document.createElement('div');
        galleryGrid.className = 'gallery-grid';
        
        // 添加所有结果
        for (let i = 0; i < results.length; i++) {
            createImageResultItem(results[i], i, galleryGrid);
        }
        
        resultsContainer.appendChild(galleryGrid);
        galleryContainer.appendChild(resultsContainer);
    }
    
    // Function to handle image upload
    function handleImageUpload(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewContainer.classList.remove('d-none');
            };
            reader.readAsDataURL(file);
        }
    }
    
    // Function to load sample image
    function loadSampleImage(path) {
        // 确保预览图片路径正确
        previewImage.src = path;
        previewContainer.classList.remove('d-none');
        
        // 记录原始路径用于调试
        previewImage.dataset.originalPath = path;
        
        // Create a File object from the sample image
        fetch(path)
            .then(res => {
                if (!res.ok) {
                    console.error('Failed to fetch sample image:', path, res.status);
                    return Promise.reject('Failed to fetch image');
                }
                return res.blob();
            })
            .then(blob => {
                const filename = getFilenameFromPath(path);
                const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
                
                // Create a new FileList-like object
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                
                // Set the file input's files
                imageUpload.files = dataTransfer.files;
            })
            .catch(err => {
                console.error('Error loading sample image:', err);
                // 如果加载失败，使用默认图片
                previewImage.src = '/static/img/no-image.png';
            });
    }
    
    // Function to perform image search
    function performImageSearch() {
        if (!previewImage.src) {
            alert('请先上传或选择图片');
            return;
        }
        
        // 获取选中的集合
        const selectedCollections = Array.from(imageCollectionCheckboxes.querySelectorAll('input:checked')).map(checkbox => checkbox.value);
        
        // Show loading indicator
        imageGalleryContainer.innerHTML = '';
        imageLoadingIndicator.classList.remove('d-none');
        
        // Get search parameters
        const topK = parseInt(imageTopKInput.value);
        const threshold = parseFloat(imageThresholdInput.value);
        const personThreshold = parseFloat(personThresholdInput.value);
        
        // Create form data
        const formData = new FormData();
        
        // If we have a file from upload
        if (imageUpload.files.length > 0) {
            formData.append('image', imageUpload.files[0]);
        } else if (previewImage.dataset.path) {
            // If we're using a sample image
            formData.append('image_path', previewImage.dataset.path);
        } else {
            // Try to get the image data from the preview
            fetch(previewImage.src)
                .then(res => res.blob())
                .then(blob => {
                    const file = new File([blob], "image.jpg", { type: "image/jpeg" });
                    formData.append('image', file);
                    continueSearch(formData);
                })
                .catch(error => {
                    console.error('获取图片数据出错:', error);
                    imageLoadingIndicator.classList.add('d-none');
                    imageGalleryContainer.innerHTML = `<div class="alert alert-danger">获取图片数据出错: ${error}</div>`;
                });
            return;
        }
        
        continueSearch(formData);
        
        function continueSearch(formData) {
            // Add parameters
            formData.append('top_k', topK);
            formData.append('threshold', threshold);
            formData.append('person_threshold', personThreshold);
            
            // 添加选中的集合
            selectedCollections.forEach(collection => {
                formData.append('collections[]', collection);
            });
            
            // Perform search
            fetch('/api/image_search', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                imageLoadingIndicator.classList.add('d-none');
                
                // Display results
                displayImageSearchResults(data);
            })
            .catch(error => {
                console.error('图片搜索出错:', error);
                imageLoadingIndicator.classList.add('d-none');
                imageGalleryContainer.innerHTML = `<div class="alert alert-danger">图片搜索出错: ${error}</div>`;
            });
        }
    }
    
    // Function to display image search results
    function displayImageSearchResults(data) {
        imageGalleryContainer.innerHTML = '';
        
        const results = data.results;
        if (!results || results.length === 0) {
            imageGalleryContainer.innerHTML = '<p class="no-results">没有找到匹配的结果</p>';
            return;
        }
        
        // 添加结果信息
        const infoText = document.createElement('div');
        infoText.className = 'results-info';
        infoText.innerHTML = `<div class="alert alert-info">
            找到 ${data.person_count || 0} 个行人，
            搜索库图片数量: ${data.record_count || 0}，
            检测耗时: ${data.detection_time ? data.detection_time.toFixed(3) : 0}秒，
            搜索耗时: ${data.search_time ? data.search_time.toFixed(3) : 0}秒，
            总耗时: ${data.total_time ? data.total_time.toFixed(3) : 0}秒
        </div>`;
        
        imageGalleryContainer.appendChild(infoText);
        
        // 处理检测结果图片
        const detectionResult = results.find(r => r.type === 'detection');
        if (detectionResult) {
            const detectionContainer = document.createElement('div');
            detectionContainer.className = 'detection-container';
            
            const detectionTitle = document.createElement('h4');
            detectionTitle.textContent = '检测结果';
            detectionContainer.appendChild(detectionTitle);
            
            const detectionImage = document.createElement('img');
            detectionImage.className = 'detection-image';
            
            // 处理图片路径
            let imagePath = detectionResult.path;
            if (imagePath.startsWith('/')) {
                detectionImage.src = `/images${imagePath}`;
            } else {
                detectionImage.src = `/images/${imagePath}`;
            }
            
            detectionImage.alt = detectionResult.label;
            
            // 点击检测图片时在模态框中显示
            detectionImage.addEventListener('click', function() {
                if (imagePath.startsWith('/')) {
                    showImageInModal(`/images${imagePath}`, detectionResult.label);
                } else {
                    showImageInModal(`/images/${imagePath}`, detectionResult.label);
                }
            });
            
            detectionContainer.appendChild(detectionImage);
            imageGalleryContainer.appendChild(detectionContainer);
        }
        
        // 创建行人结果容器
        const personsContainer = document.createElement('div');
        personsContainer.className = 'persons-container';
        
        // 按行人ID分组结果
        const personGroups = {};
        
        // 获取行人图片
        const personResults = results.filter(r => r.type === 'person');
        personResults.forEach(person => {
            personGroups[person.person_id] = {
                person: person,
                results: []
            };
        });
        
        // 获取每个行人的搜索结果
        const searchResults = results.filter(r => r.type === 'result');
        searchResults.forEach(result => {
            if (personGroups[result.person_id]) {
                personGroups[result.person_id].results.push(result);
            }
        });
        
        // 显示每个行人的结果
        Object.values(personGroups).forEach(group => {
            const personContainer = document.createElement('div');
            personContainer.className = 'person-container';
            
            // 行人标题
            const personTitle = document.createElement('h4');
            personTitle.textContent = group.person.label;
            personTitle.style.color = `rgb(${group.person.color.join(',')})`;
            personContainer.appendChild(personTitle);
            
            // 行人图片
            const personImage = document.createElement('img');
            personImage.className = 'person-image';
            
            // 处理图片路径
            let personImagePath = group.person.path;
            if (personImagePath.startsWith('/')) {
                personImage.src = `/images${personImagePath}`;
            } else {
                personImage.src = `/images/${personImagePath}`;
            }
            
            personImage.alt = group.person.label;
            
            // 点击行人图片时在模态框中显示
            personImage.addEventListener('click', function() {
                if (personImagePath.startsWith('/')) {
                    showImageInModal(`/images${personImagePath}`, group.person.label);
                } else {
                    showImageInModal(`/images/${personImagePath}`, group.person.label);
                }
            });
            
            personContainer.appendChild(personImage);
            
            // 行人搜索结果
            if (group.results.length > 0) {
                const resultsTitle = document.createElement('h5');
                resultsTitle.textContent = `搜索结果 (${group.results.length})`;
                personContainer.appendChild(resultsTitle);
                
                const personResults = document.createElement('div');
                personResults.className = 'person-results';
                
                // 添加所有结果
                group.results.forEach(result => {
                    createImageResultItem(result, result.person_id, personResults);
                });
                
                personContainer.appendChild(personResults);
            } else {
                const noResults = document.createElement('p');
                noResults.textContent = '没有找到匹配的结果';
                personContainer.appendChild(noResults);
            }
            
            personsContainer.appendChild(personContainer);
        });
        
        imageGalleryContainer.appendChild(personsContainer);
    }
    
    // Function to clear text search
    function clearTextSearch() {
        queryInput.value = '';
        galleryContainer.innerHTML = '';
    }
    
    // Function to clear image search
    function clearImageSearch() {
        imageUpload.value = '';
        previewContainer.classList.add('d-none');
        imageGalleryContainer.innerHTML = '';
    }
    
    // Function to show image in modal
    function showImageInModal(path, caption) {
        console.log("显示图片:", path);
        
        // 设置加载状态
        modalImage.classList.remove('loaded');
        modalImage.classList.add('loading');
        
        // 设置图片路径
        modalImage.src = path;
        modalImage.dataset.cacheKey = path;
        
        // 获取文件名
        const filename = path.split('/').pop();
        
        // 设置标题和说明
        modalCaption.textContent = caption || filename;
        modalTitle.textContent = filename;
        
        // 添加加载指示器
        const loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'spinner-border text-primary position-absolute';
        loadingIndicator.style.top = '50%';
        loadingIndicator.style.left = '50%';
        loadingIndicator.style.transform = 'translate(-50%, -50%)';
        
        // 清除之前的内容
        const modalBody = document.querySelector('.modal-body');
        while (modalBody.firstChild) {
            if (modalBody.firstChild !== modalImage) {
                modalBody.removeChild(modalBody.firstChild);
            } else {
                break;
            }
        }
        
        // 添加加载指示器
        modalBody.appendChild(loadingIndicator);
        
        // 监听图片加载完成事件
        modalImage.onload = function() {
            console.log("图片加载完成:", path);
            // 移除加载类
            modalImage.classList.remove('loading');
            modalImage.classList.add('loaded');
            
            // 移除加载指示器
            if (loadingIndicator.parentNode) {
                loadingIndicator.parentNode.removeChild(loadingIndicator);
            }
        };
        
        // 处理加载错误
        modalImage.onerror = function() {
            console.error('图片加载失败:', path);
            
            // 尝试修复路径
            if (path.includes('/images/')) {
                // 尝试不同的路径格式
                let fixedPath;
                if (path.includes('/images//')) {
                    // 修复双斜杠问题
                    fixedPath = path.replace('/images//', '/images/');
                } else {
                    // 尝试直接使用文件名
                    fixedPath = `/images/${filename}`;
                }
                
                console.log("尝试修复路径:", fixedPath);
                modalImage.src = fixedPath;
                return;
            }
            
            // 移除加载指示器
            if (loadingIndicator.parentNode) {
                loadingIndicator.parentNode.removeChild(loadingIndicator);
            }
            
            // 显示错误信息
            const errorMsg = document.createElement('div');
            errorMsg.className = 'alert alert-danger mt-3';
            errorMsg.textContent = '图片加载失败';
            modalBody.appendChild(errorMsg);
        };
        
        imageModal.show();
    }
    
    // Helper function to get filename from path
    function getFilenameFromPath(path) {
        return path.split('\\').pop().split('/').pop();
    }
    
    // Function to lazy load images
    function lazyLoadImage(imgElement) {
        const fullSrc = imgElement.dataset.src;
        
        if (!fullSrc) return;
        
        const img = new Image();
        img.src = fullSrc;
        
        img.onload = function() {
            imgElement.src = fullSrc;
            imgElement.classList.add('loaded');
        };
    }
    
    // Function to preload an image
    function preloadImage(src) {
        const img = new Image();
        img.src = src;
    }
    
    // Function to create image result item
    function createImageResultItem(result, index, galleryContainer) {
        const galleryItem = document.createElement('div');
        galleryItem.className = 'gallery-item';
        
        // Image container
        const imgContainer = document.createElement('div');
        imgContainer.className = 'img-container';
        
        // 使用预览图而不是缩略图
        const previewImg = document.createElement('img');
        previewImg.className = 'preview';
        
        // 处理图片路径
        let imagePath = result.path || '';
        
        // 确保路径正确处理 - 现在后端只返回文件名，不返回完整路径
        if (imagePath) {
            // 检查是否已经是完整URL
            if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) {
                previewImg.src = imagePath;
            } 
            // 检查是否已经是以/开头的路径
            else if (imagePath.startsWith('/')) {
                previewImg.src = imagePath;
            } 
            // 否则作为相对路径处理
            else {
                previewImg.src = `/images/${imagePath}`;
            }
        } else {
            // 如果没有有效路径，使用默认图片
            previewImg.src = '/static/img/no-image.png';
        }
        
        // 存储原始路径用于调试
        previewImg.dataset.originalPath = imagePath;
        
        previewImg.alt = result.label || `Result ${index + 1}`;
        previewImg.loading = 'lazy';
        
        // 存储原始路径以便后续更新
        if (result.original_path) {
            const originalPath = result.original_path;
            // 确保缓存键是文件名而不是完整路径
            const cacheKey = originalPath.includes('/') ? originalPath.split('/').pop() : originalPath;
            previewImg.dataset.cacheKey = cacheKey;
            
            // 添加到待处理集合
            pendingImages.add(cacheKey);
        }
        
        imgContainer.appendChild(previewImg);
        
        // 添加点击事件
        imgContainer.addEventListener('click', function() {
            const fullPath = result.original_path || result.path;
            let displayPath;
            
            // 确保路径格式正确
            if (fullPath.startsWith('http://') || fullPath.startsWith('https://')) {
                displayPath = fullPath;
            } else if (fullPath.startsWith('/')) {
                displayPath = fullPath;
            } else {
                displayPath = `/images/${fullPath}`;
            }
            
            showImageInModal(displayPath, result.label || `Result ${index + 1}`);
        });
        
        galleryItem.appendChild(imgContainer);
        
        // 添加标签
        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = result.label || `Result ${index + 1}`;
        galleryItem.appendChild(label);
        
        // 添加分数
        if (result.score !== undefined) {
            const score = document.createElement('div');
            score.className = 'score';
            score.textContent = `Score: ${result.score.toFixed(3)}`;
            galleryItem.appendChild(score);
        }
        
        galleryContainer.appendChild(galleryItem);
    }
    
    // Function to initialize lazy loading for all images
    function initLazyLoading() {
        const lazyImages = document.querySelectorAll('img[data-src]');
        lazyImages.forEach(img => {
            lazyLoadImage(img);
        });
    }
    
    // Register images functionality
    const registerBtn = document.getElementById('register-btn');
    const imageDirInput = document.getElementById('image-dir');
    const categoryInput = document.getElementById('category');
    const registerStatus = document.getElementById('register-status');
    
    registerBtn.addEventListener('click', function() {
        const imageDir = imageDirInput.value.trim();
        const category = categoryInput.value.trim();
        
        if (!imageDir) {
            alert('请输入图片目录路径');
            return;
        }
        
        registerStatus.style.display = 'block';
        registerStatus.textContent = '正在注册图片，请稍候...';
        registerStatus.className = 'alert alert-info';
        
        // Register images API call
        fetch('/api/register_images', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_dir: imageDir,
                category: category
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                registerStatus.textContent = data.message;
                registerStatus.className = 'alert alert-success';
                
                // Poll for status updates
                const statusInterval = setInterval(() => {
                    fetch('/api/register_status')
                        .then(response => response.json())
                        .then(statusData => {
                            registerStatus.textContent = `当前记录数: ${statusData.count}`;
                            
                            // Update record count displays
                            if (recordCountDisplay) {
                                recordCountDisplay.value = `图片数量: ${statusData.count}`;
                            }
                        })
                        .catch(error => {
                            console.error('Error checking status:', error);
                        });
                }, 2000);
                
                // Stop polling after 30 seconds
                setTimeout(() => {
                    clearInterval(statusInterval);
                }, 30000);
            } else {
                registerStatus.textContent = data.error || '注册失败';
                registerStatus.className = 'alert alert-danger';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            registerStatus.textContent = '注册出错，请重试';
            registerStatus.className = 'alert alert-danger';
        });
    });
    
    // 初始化集合复选框
    function initCollections() {
        fetch('/api/get_collections')
            .then(response => response.json())
            .then(data => {
                // 清空复选框容器
                document.getElementById('collection-checkboxes').innerHTML = '';
                document.getElementById('image-collection-checkboxes').innerHTML = '';
                
                // 添加集合复选框
                data.collections.forEach(collection => {
                    // 为文本搜索创建复选框
                    const checkboxDiv = document.createElement('div');
                    checkboxDiv.className = 'form-check form-check-inline me-2';
                    
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.className = 'form-check-input collection-checkbox';
                    checkbox.value = collection;
                    checkbox.id = `collection-${collection}`;
                    
                    // 设置默认选中状态
                    if (collection === 'zkteco_xm' || collection === data.current) {
                        checkbox.checked = true;
                    }
                    
                    const label = document.createElement('label');
                    label.className = 'form-check-label';
                    label.htmlFor = `collection-${collection}`;
                    label.textContent = collection;
                    
                    checkboxDiv.appendChild(checkbox);
                    checkboxDiv.appendChild(label);
                    
                    // 添加到文本搜索的复选框容器
                    document.getElementById('collection-checkboxes').appendChild(checkboxDiv);
                    
                    // 为图片搜索创建复选框（克隆上面的元素）
                    const imageCheckboxDiv = checkboxDiv.cloneNode(true);
                    imageCheckboxDiv.querySelector('input').id = `image-collection-${collection}`;
                    imageCheckboxDiv.querySelector('label').htmlFor = `image-collection-${collection}`;
                    imageCheckboxDiv.querySelector('input').className = 'form-check-input image-collection-checkbox';
                    
                    // 添加到图片搜索的复选框容器
                    document.getElementById('image-collection-checkboxes').appendChild(imageCheckboxDiv);
                });
                
                // 更新记录数量显示
                updateRecordCount();
            })
            .catch(error => {
                console.error('获取集合列表失败:', error);
            });
    }
    
    // 切换集合
    function switchCollection(collectionNames) {
        // 如果是单个集合名称，转换为数组
        if (!Array.isArray(collectionNames)) {
            collectionNames = [collectionNames];
        }
        
        // 确保至少选择了一个集合
        if (collectionNames.length === 0) {
            alert('请至少选择一个集合');
            return;
        }
        
        console.log(`切换到集合: ${collectionNames.join(', ')}`);
        
        // 发送请求切换集合
        fetch('/api/set_collection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                collection_name: collectionNames[0], // 使用第一个集合作为主集合
                collections: collectionNames // 传递所有选中的集合
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                console.log(`成功切换到集合: ${collectionNames.join(', ')}`);
                
                // 直接使用返回的记录数量
                if (data.record_count !== undefined) {
                    console.log(`从API返回的记录数量: ${data.record_count}`);
                    recordCountDisplay.value = `注册图片数量: ${data.record_count}`;
                } else {
                    // 如果API没有返回记录数量，则调用updateRecordCount
                    console.log('API没有返回记录数量，调用updateRecordCount()');
                    updateRecordCount();
                }
                
                // 清空搜索结果
                clearTextSearch();
                clearImageSearch();
            } else {
                console.error(`切换集合失败: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('切换集合失败:', error);
        });
    }
    
    // 更新记录数量显示
    function updateRecordCount() {
        console.log("正在更新记录数量显示...");
        
        // 添加加载状态提示
        recordCountDisplay.value = "正在加载记录数量...";
        
        fetch('/api/record_count')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data && data.count !== undefined) {
                    console.log(`获取到记录数量: ${data.count}`);
                    // 确保显示为0而不是NULL，当count为0或null时
                    const count = data.count !== null ? data.count : 0;
                    recordCountDisplay.value = `注册图片数量: ${count}`;
                } else {
                    console.error('获取记录数量失败: 返回数据无效', data);
                    recordCountDisplay.value = "注册图片数量: 0";
                    
                    // 延迟1秒后重试一次
                    setTimeout(updateRecordCount, 1000);
                }
            })
            .catch(error => {
                console.error('获取记录数量失败:', error);
                recordCountDisplay.value = "注册图片数量: 0";
                
                // 延迟2秒后重试一次
                setTimeout(updateRecordCount, 2000);
            });
    }
});
