'use strict';

angular.module('images').controller('ScoreCtrl', function($scope, $http) {

  $scope.models = "forest,meadow,mountain,water,urban,desert";

  $scope.score = function() {
    $http.get("/score", {
      params: {
        url: $scope.url,
        models: $scope.models 
      }
    }).then(function success(response) {
      $scope.result = response.data;
      $scope.$error = undefined;
    }, function error(response) {
      $scope.result = undefined;
      $scope.$error = response.data;
    });
  };
});
