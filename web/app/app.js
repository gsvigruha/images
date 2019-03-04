'use strict';

// Declare app level module which depends on views, and components
angular.module('images', [
  'ngRoute',
])
.config(function ($routeProvider, $locationProvider) {
    $routeProvider
      .when('/', {
        templateUrl: 'score.html',
        controller: 'ScoreCtrl',
      });
});
