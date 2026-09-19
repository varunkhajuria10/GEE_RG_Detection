// Supervised Classification with Satellite Embeddings
// ====================================================
// This Google Earth Engine script performs a supervised classification
// of rock glacier (RG) versus non-rock glacier (NONRG) areas using the
// AlphaEarth satellite embedding collection. Before running the script
// in the Earth Engine Code Editor, draw the study area polygon and the
// RG/NONRG training polygons and rename them to `geometry`, `RG`, and
// `NONRG` respectively.

// This script contains only the Google Earth Engine workflow used for embedding extraction,
// classifier training, validation, and regional classification. Additional analyses reported
// in the manuscript, including sensitivity tests, bootstrap/OOB uncertainty assessment,
// post-processing, aggregation tests, and the conventional multi-sensor benchmark,
// were conducted independently using separate scripts. 
//
// Author: Varun Khajuria
// Last updated: 11/11/2025
//
// ----------------------------------------------------
// 1. Region of interest
// ----------------------------------------------------

Map.centerObject(geometry, 12);

// ----------------------------------------------------
// 2. Sentinel-2 composite for visual reference
// ----------------------------------------------------

var year = 2024;

// Sentinel-2 date range (2-month window in summer)
var s2StartDate = ee.Date.fromYMD(year, 7, 1);
var s2EndDate   = s2StartDate.advance(2, 'month');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var filteredS2 = s2
  .filter(ee.Filter.date(s2StartDate, s2EndDate))
  .filter(ee.Filter.bounds(geometry));

// Cloud Score+ collection for S2
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

// Link Cloud Score+ to Sentinel-2 collection
var filteredS2WithCs = filteredS2.linkCollection(csPlus, ['cs']);

function maskLowQA(image) {
  var qaBand = 'cs';
  var clearThreshold = 0.6;
  var mask = image.select(qaBand).gte(clearThreshold);
  return image.updateMask(mask).select('B.*');
}

var filteredS2Masked = filteredS2WithCs.map(maskLowQA);

// Median composite
var composite = filteredS2Masked.median();

// Display S2 composite in RGB
var rgbVis = {min: 300, max: 4000, bands: ['B4', 'B3', 'B2']};
Map.addLayer(composite.clip(geometry), rgbVis, 'S2 Composite (RGB)', false);

// ----------------------------------------------------
// 3. Prepare training polygons (RG vs NONRG)
// ----------------------------------------------------

// RG and NONRG are FeatureCollections drawn in the GEE editor

// Add class = 1 (RG)
var rg_classified = RG.map(function(feature) {
  return feature.set('class', 1);
});

// Add class = 0 (NONRG)
var nonrg_classified = NONRG.map(function(feature) {
  return feature.set('class', 0);
});

// Merge into a single ground control point collection
var gcps = rg_classified.merge(nonrg_classified);

// ----------------------------------------------------
// 4. Load satellite embeddings (AlphaEarth V1 annual)
// ----------------------------------------------------

var embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');

// Annual date range for embeddings (full year)
var embeddingsStartDate = ee.Date.fromYMD(year, 1, 1);
var embeddingsEndDate   = embeddingsStartDate.advance(1, 'year');

var embeddingsFiltered = embeddings
  .filter(ee.Filter.date(embeddingsStartDate, embeddingsEndDate))
  .filter(ee.Filter.bounds(geometry));

// Mosaic (should be a single annual image anyway)
var embeddingsImage = embeddingsFiltered.mosaic();

// ----------------------------------------------------
// 5. Train / validation split (70 / 30)
// ----------------------------------------------------

var gcpsWithRandom = gcps.randomColumn('random');

var trainingSet = gcpsWithRandom.filter(ee.Filter.lt('random', 0.7)); 
var validationSet = gcpsWithRandom.filter(ee.Filter.gte('random', 0.7));

// Reduce embeddings over polygons (mean per polygon)
var training = embeddingsImage.reduceRegions({
  collection: trainingSet,
  reducer: ee.Reducer.mean(),
  scale: 10,
  tileScale: 16
});

var validation = embeddingsImage.reduceRegions({
  collection: validationSet,
  reducer: ee.Reducer.mean(),
  scale: 10,
  tileScale: 16
});

// Filter out polygons that returned null in A00 (very tiny polygons etc.)
training   = training.filter(ee.Filter.notNull(['A00']));
validation = validation.filter(ee.Filter.notNull(['A00']));

//sanity checks: counts per class
print('Training counts by class:', trainingSet.aggregate_histogram('class'));
print('Validation counts by class:', validationSet.aggregate_histogram('class'));

// ----------------------------------------------------
// 6. Define and train classifiers
// ----------------------------------------------------

// Use all embedding bands
var embeddingBands = embeddingsImage.bandNames();

// 1) K-Nearest Neighbors (KNN)
var knnClassifier = ee.Classifier.smileKNN(5, 'AUTO', 'EUCLIDEAN').train({
  features: training,
  classProperty: 'class',
  inputProperties: embeddingBands
});

// 2) Random Forest (RF)
var rfClassifier = ee.Classifier.smileRandomForest(
  100,   // numberOfTrees
  null,  // variablesPerSplit
  1,     // minLeafPopulation
  0.7,   // bagFraction
  null,  // maxNodes
  0      // seed
).train({
  features: training,
  classProperty: 'class',
  inputProperties: embeddingBands
});

// 3) Support Vector Machine (SVM)
var svmClassifier = ee.Classifier.libsvm({
  kernelType: 'RBF',
  gamma: 0.5,
  cost: 10
}).train({
  features: training,
  classProperty: 'class',
  inputProperties: embeddingBands
});

// 4) Gradient Tree Boost (GTB)
var gtbClassifier = ee.Classifier.smileGradientTreeBoost({
  numberOfTrees: 50,
  shrinkage: 0.1,
  samplingRate: 0.7,
  maxNodes: 1000,
  loss: 'Logistic'
}).train({
  features: training,
  classProperty: 'class',
  inputProperties: embeddingBands
});

// ----------------------------------------------------
// 7. Classify the full embedding mosaic
// ----------------------------------------------------

var classifiedKNN = embeddingsImage.classify(knnClassifier);
var classifiedRF  = embeddingsImage.classify(rfClassifier);
var classifiedSVM = embeddingsImage.classify(svmClassifier);
var classifiedGTB = embeddingsImage.classify(gtbClassifier);

// Visualization palette: 0 = NONRG (red), 1 = RG (green)
var palette = ['red', 'green'];
var viz_class = {
  min: 0,
  max: 1,
  palette: palette
};

Map.addLayer(classifiedKNN.clip(geometry), viz_class, 'Classified (KNN)', false);
Map.addLayer(classifiedRF.clip(geometry),  viz_class, 'Classified (Random Forest)', true);
Map.addLayer(classifiedSVM.clip(geometry), viz_class, 'Classified (SVM)', false);
Map.addLayer(classifiedGTB.clip(geometry), viz_class, 'Classified (Gradient Tree Boost)', false);

// Visualise embeddings as pseudo-RGB
var embeddingsRGB = {
  bands: ['A01', 'A16', 'A09'],
  min: -0.3,
  max: 0.3
};
Map.addLayer(embeddingsImage.clip(geometry), embeddingsRGB, 'Embeddings (RGB of selected bands)', false);

// ----------------------------------------------------
// 8. Enhanced accuracy assessment (with F1 + MCC)
// ----------------------------------------------------

function computeDetailedMetrics(validationData, classifier, classifierName) {
  var validated       = validationData.classify(classifier);
  var confusionMatrix = validated.errorMatrix('class', 'classification');

  // Basic metrics
  var overallAccuracy   = confusionMatrix.accuracy();
  var kappa             = confusionMatrix.kappa();
  var producersAccuracy = confusionMatrix.producersAccuracy();
  var usersAccuracy     = confusionMatrix.consumersAccuracy();

  // Convert confusion matrix to ee.Array
  var matrixArray = confusionMatrix.array();

  var tn = matrixArray.get([0, 0]);
  var fp = matrixArray.get([0, 1]);
  var fn = matrixArray.get([1, 0]);
  var tp = matrixArray.get([1, 1]);

  // ---- Class 0 (NONRG) metrics ----
  var precision0 = tn.divide(tn.add(fn));  // predicted NONRG that are correct
  var recall0    = tn.divide(tn.add(fp));  // actual NONRG correctly predicted
  var f1_0       = precision0.multiply(recall0).multiply(2)
                     .divide(precision0.add(recall0));

  // ---- Class 1 (RG) metrics (standard definitions) ----
  var precision1 = tp.divide(tp.add(fp));
  var recall1    = tp.divide(tp.add(fn));
  var f1_1       = precision1.multiply(recall1).multiply(2)
                     .divide(precision1.add(recall1));

  // ---- Matthews Correlation Coefficient (MCC) ----
  var mccDen = tp.add(fp)
    .multiply(tp.add(fn))
    .multiply(tn.add(fp))
    .multiply(tn.add(fn))
    .sqrt();
  var mcc = tp.multiply(tn).subtract(fp.multiply(fn)).divide(mccDen);

  // Print detailed results
  print('=== ' + classifierName + ' Detailed Metrics ===');
  print('Confusion Matrix:', confusionMatrix);
  print('Overall Accuracy:', overallAccuracy);
  print('Kappa Coefficient:', kappa);
  print('Matthews Correlation Coefficient:', mcc);

  print('--- Class 0 (NONRG) ---');
  print('Precision:', precision0);
  print('Recall (Sensitivity):', recall0);
  print('F1-Score:', f1_0);

  print('--- Class 1 (RG) ---');
  print('Precision:', precision1);
  print('Recall (Sensitivity):', recall1);
  print('F1-Score:', f1_1);

  print('Producer\'s Accuracy (per class):', producersAccuracy);
  print('User\'s Accuracy (per class):', usersAccuracy);

  return {
    name: classifierName,
    accuracy: overallAccuracy,
    kappa: kappa,
    mcc: mcc,
    f1_rg: f1_1,
    precision_rg: precision1,
    recall_rg: recall1
  };
}

print('COMPREHENSIVE ACCURACY ASSESSMENT');
print('=================================');

var knnMetrics = computeDetailedMetrics(validation, knnClassifier, 'K-Nearest Neighbours (k=5)');
var rfMetrics  = computeDetailedMetrics(validation, rfClassifier,  'Random Forest');
var svmMetrics = computeDetailedMetrics(validation, svmClassifier, 'Support Vector Machine');
var gtbMetrics = computeDetailedMetrics(validation, gtbClassifier, 'Gradient Tree Boost');

// ----------------------------------------------------
// 9. Classifier comparison chart
// ----------------------------------------------------

var comparisonData = ee.FeatureCollection([
  ee.Feature(null, {
    'Classifier': 'KNN',
    'Accuracy': knnMetrics.accuracy,
    'Kappa': knnMetrics.kappa,
    'MCC': knnMetrics.mcc,
    'F1-Score': knnMetrics.f1_rg
  }),
  ee.Feature(null, {
    'Classifier': 'Random Forest',
    'Accuracy': rfMetrics.accuracy,
    'Kappa': rfMetrics.kappa,
    'MCC': rfMetrics.mcc,
    'F1-Score': rfMetrics.f1_rg
  }),
  ee.Feature(null, {
    'Classifier': 'SVM',
    'Accuracy': svmMetrics.accuracy,
    'Kappa': svmMetrics.kappa,
    'MCC': svmMetrics.mcc,
    'F1-Score': svmMetrics.f1_rg
  }),
  ee.Feature(null, {
    'Classifier': 'Gradient Tree Boost',
    'Accuracy': gtbMetrics.accuracy,
    'Kappa': gtbMetrics.kappa,
    'MCC': gtbMetrics.mcc,
    'F1-Score': gtbMetrics.f1_rg
  })
]);

var comparisonChart = ui.Chart.feature.byFeature({
  features: comparisonData,
  xProperty: 'Classifier',
  yProperties: ['Accuracy', 'Kappa', 'MCC', 'F1-Score']
}).setChartType('ColumnChart')
  .setOptions({
    title: 'Classifier Performance Comparison',
    hAxis: {title: 'Classifiers'},
    vAxis: {title: 'Score', viewWindow: {min: 0, max: 1}},
    legend: {position: 'top'}
  });

print('Classifier Comparison Chart:', comparisonChart);

// ----------------------------------------------------
// 10. Area calculation (RG area by classifier)
// ----------------------------------------------------

var pixelArea = ee.Image.pixelArea();

function computeAreaSqKm(classifiedImage, label) {
  var rgArea = classifiedImage.eq(1)  // class 1 = RG
    .multiply(pixelArea)
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: geometry,
      scale: 10,
      maxPixels: 1e13
    });
  var areaSqM  = ee.Number(rgArea.get('classification'));
  var areaSqKm = areaSqM.divide(1e6);
  print(label + ' RG Area (sq m):', areaSqM);
  print(label + ' RG Area (sq km):', areaSqKm);
  return areaSqKm;
}

print('=== ROCK GLACIER AREA ESTIMATION ===');
computeAreaSqKm(classifiedKNN, 'KNN');
computeAreaSqKm(classifiedRF,  'Random Forest');
computeAreaSqKm(classifiedSVM, 'SVM');
computeAreaSqKm(classifiedGTB, 'Gradient Tree Boost');

// ----------------------------------------------------
// 11. Original simple accuracy matrices
// ----------------------------------------------------

var validatedKNN = validation.classify(knnClassifier);
var cmKNN = validatedKNN.errorMatrix('class', 'classification');
print('KNN Validation Error Matrix', cmKNN);
print('KNN Overall Accuracy', cmKNN.accuracy());
print('KNN Kappa Coefficient', cmKNN.kappa());

var validatedRF = validation.classify(rfClassifier);
var cmRF = validatedRF.errorMatrix('class', 'classification');
print('Random Forest Validation Error Matrix', cmRF);
print('Random Forest Overall Accuracy', cmRF.accuracy());
print('Random Forest Kappa Coefficient', cmRF.kappa());

var validatedSVM = validation.classify(svmClassifier);
var cmSVM = validatedSVM.errorMatrix('class', 'classification');
print('SVM Validation Error Matrix', cmSVM);
print('SVM Overall Accuracy', cmSVM.accuracy());
print('SVM Kappa Coefficient', cmSVM.kappa());

var validatedGTB = validation.classify(gtbClassifier);
var cmGTB = validatedGTB.errorMatrix('class', 'classification');
print('Gradient Tree Boost Validation Error Matrix', cmGTB);
print('Gradient Tree Boost Overall Accuracy', cmGTB.accuracy());
print('Gradient Tree Boost Kappa Coefficient', cmGTB.kappa());

// ----------------------------------------------------
// 12. Legend
// ----------------------------------------------------

var legend = ui.Panel({
  style: {position: 'bottom-right', padding: '8px 15px'}
});

var makeRow = function(color, name) {
  var colorBox = ui.Label({
    style: {
      color: '#ffffff',
      backgroundColor: color,
      padding: '10px',
      margin: '0 0 4px 0'
    }
  });
  var description = ui.Label({
    value: name,
    style: {margin: '0 0 4px 6px'}
  });
  return ui.Panel({
    widgets: [colorBox, description],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
};

var title = ui.Label({
  value: 'Legend',
  style: {
    fontWeight: 'bold',
    fontSize: '16px',
    margin: '0 0 4px 0'
  }
});

legend.add(title);
legend.add(makeRow('red',   'NONRG'));
legend.add(makeRow('green', 'RG'));

Map.add(legend);

// ----------------------------------------------------
// 13. Exports
// ----------------------------------------------------

Export.image.toDrive({
  image: classifiedKNN,
  description: 'RockGlacier_Classification_KNN',
  scale: 10,
  region: geometry,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: classifiedRF,
  description: 'RockGlacier_Classification_RF',
  scale: 10,
  region: geometry,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: classifiedSVM,
  description: 'RockGlacier_Classification_SVM',
  scale: 10,
  region: geometry,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: classifiedGTB,
  description: 'RockGlacier_Classification_GTB',
  scale: 10,
  region: geometry,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: composite,
  description: 'Sentinel2Composite',
  scale: 10,
  region: geometry,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: embeddingsImage,
  description: 'RockGlacier_Embeddings',
  scale: 10,
  region: geometry,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

Export.table.toDrive({
  collection: training,
  description: 'Training_Data_Reference',
  fileFormat: 'CSV'
});

// ----------------------------------------------------
// 14. Feature importance (single-band accuracy, multi-classifier)
// ----------------------------------------------------

print('=== FEATURE IMPORTANCE ANALYSIS (RF, SVM, KNN, GTB) ===');

var bandNames = embeddingsImage.bandNames().slice(0, 64); // use all embedding bands

var importanceResults = bandNames.map(function(bandName) {
  var props = [bandName];

  // RF
  var simpleRF = ee.Classifier.smileRandomForest(100).train({
    features: training,
    classProperty: 'class',
    inputProperties: props
  });
  var valRF = validation.classify(simpleRF);
  var accRF = valRF.errorMatrix('class', 'classification').accuracy();

  // SVM
  var simpleSVM = ee.Classifier.libsvm({
    kernelType: 'RBF',
    gamma: 0.5,
    cost: 10
  }).train({
    features: training,
    classProperty: 'class',
    inputProperties: props
  });
  var valSVM = validation.classify(simpleSVM);
  var accSVM = valSVM.errorMatrix('class', 'classification').accuracy();

  // KNN
  var simpleKNN = ee.Classifier.smileKNN(5, 'AUTO', 'EUCLIDEAN').train({
    features: training,
    classProperty: 'class',
    inputProperties: props
  });
  var valKNN = validation.classify(simpleKNN);
  var accKNN = valKNN.errorMatrix('class', 'classification').accuracy();

  // GTB
  var simpleGTB = ee.Classifier.smileGradientTreeBoost({
    numberOfTrees: 50,
    shrinkage: 0.1,
    samplingRate: 0.7,
    maxNodes: 1000,
    loss: 'Logistic'
  }).train({
    features: training,
    classProperty: 'class',
    inputProperties: props
  });
  var valGTB = validation.classify(simpleGTB);
  var accGTB = valGTB.errorMatrix('class', 'classification').accuracy();

  // Mean accuracy across classifiers (simple ensemble importance)
  var accMean = accRF.add(accSVM).add(accKNN).add(accGTB).divide(4);

  return ee.Feature(null, {
    'Band': bandName,
    'Accuracy_RF': accRF,
    'Accuracy_SVM': accSVM,
    'Accuracy_KNN': accKNN,
    'Accuracy_GTB': accGTB,
    'Accuracy_Mean': accMean
  });
});

var importanceFC = ee.FeatureCollection(importanceResults);

var importanceChart = ui.Chart.feature.byFeature({
  features: importanceFC,
  xProperty: 'Band',
  yProperties: ['Accuracy_RF', 'Accuracy_SVM', 'Accuracy_KNN', 'Accuracy_GTB', 'Accuracy_Mean']
}).setOptions({
  title: 'Feature Importance (Single-Band Accuracy for Multiple Classifiers)',
  hAxis: {title: 'Embedding Bands', slantedText: true, slantedTextAngle: 45},
  vAxis: {title: 'Accuracy', viewWindow: {min: 0, max: 1}},
  legend: {position: 'top'}
});

print('Feature Importance Chart (multi-classifier):', importanceChart);

// ----------------------------------------------------
// 15. Final messages
// ----------------------------------------------------

print('=== PROCESSING COMPLETE ===');
print('All classifiers (KNN, RF, SVM, GTB) trained and evaluated successfully!');
print('Enhanced metrics include: Precision, Recall, F1-Score, MCC, area calculations, and multi-classifier feature importance.');
