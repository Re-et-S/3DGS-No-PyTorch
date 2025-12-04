#pragma once
#include "ColmapLoader.h"
#include "ImageIO.h"

struct TrainingItem {
    // Using pointer for lightweight assignability (as discussed)
    const TrainingView* view;
    std::vector<float> gt_image; 
};

class Dataset {
public:
    Dataset(const std::string& sparse, const std::string& images) 
        : loader_(sparse, images) 
    {
        // 1. Initialize Loader internally
        loader_.loadAll();
        loader_.buildTrainingViews(0.01f, 100.0f);
        
        // 2. Cache useful metadata
        auto [w, h] = loader_.getMaxDimensions();
        max_w_ = w;
        max_h_ = h;
    }

    // --- Data Access ---

    size_t size() const {
        return loader_.getTrainingViews().size();
    }

    TrainingItem get_item(int index) {
        TrainingItem item;
        
        // 1. Get View Metadata (Pointer to stable memory in loader)
        item.view = &loader_.getView(index);

        // 2. Load Image Data (Lazy Loading)

        // Construct path using loader's base path
        std::string path = loader_.getImagePath() + item.view->image_name;
        
        // Call the free function from ImageIO
        item.gt_image = load_image_planar(path, max_w_, max_h_);

        return item;
    }

    std::pair<int, int> getMaxDimensions() const {
        return {max_w_, max_h_};
    }
    
    // Expose points for Scene initialization
    const std::vector<ColmapPoint3D>& getPoints() const {
        return loader_.getPoints();
    }

    
private:
    ColmapLoader loader_;
    int max_w_, max_h_;
};
