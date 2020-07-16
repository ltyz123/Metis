/*!
 * AdminLTE v3.0.0-alpha.2 (https://adminlte.io)
 * Copyright 2014-2018 Abdullah Almsaeed <abdullah@almsaeedstudio.com>
 * Licensed under MIT (https://github.com/almasaeed2010/AdminLTE/blob/master/LICENSE)
 */
(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
  typeof define === 'function' && define.amd ? define(['exports'], factory) :
  (factory((global.adminlte = {})));
}(this, (function (exports) { 'use strict';

var _typeof = typeof Symbol === "function" && typeof Symbol.iterator === "symbol" ? function (obj) {
  return typeof obj;
} : function (obj) {
  return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
};


var classCallCheck = function (instance, Constructor) {
  if (!(instance instanceof Constructor)) {
    throw new TypeError("Cannot call a class as a function");
  }
};

/**
 * --------------------------------------------
 * AdminLTE Layout.js
 * License MIT
 * --------------------------------------------
 */

var Layout = function ($) {
  /**
   * Constants
   * ====================================================
   */

  var NAME = 'Layout';
  var DATA_KEY = 'lte.layout';
  var JQUERY_NO_CONFLICT = $.fn[NAME];

  var Selector = {
    SIDEBAR: '.main-sidebar',
    HEADER: '.main-header',
    CONTENT: '.content-wrapper',
    CONTENT_HEADER: '.content-header',
    WRAPPER: '.wrapper',
    CONTROL_SIDEBAR: '.control-sidebar',
    LAYOUT_FIXED: '.layout-fixed',
    FOOTER: '.main-footer'
  };

  var ClassName = {
    HOLD: 'hold-transition',
    SIDEBAR: 'main-sidebar',
    LAYOUT_FIXED: 'layout-fixed'

    /**
     * Class Definition
     * ====================================================
     */

  };
  var Layout = function () {
    function Layout(element) {
      classCallCheck(this, Layout);

      this._element = element;

      this._init();
    }

    // Public

    Layout.prototype.fixLayoutHeight = function fixLayoutHeight() {
      var heights = {
        window: $(window).height(),
        header: $(Selector.HEADER).outerHeight(),
        footer: $(Selector.FOOTER).outerHeight(),
        sidebar: $(Selector.SIDEBAR).height()
      };
      var max = this._max(heights);

      $(Selector.CONTENT).css('min-height', max - heights.header);
      $(Selector.SIDEBAR).css('min-height', max - heights.header);
    };

    // Private

    Layout.prototype._init = function _init() {
      var _this = this;

      // Enable transitions
      $('body').removeClass(ClassName.HOLD);

      // Activate layout height watcher
      this.fixLayoutHeight();
      $(Selector.SIDEBAR).on('collapsed.lte.treeview expanded.lte.treeview collapsed.lte.pushmenu expanded.lte.pushmenu', function () {
        _this.fixLayoutHeight();
      });

      $(window).resize(function () {
        _this.fixLayoutHeight();
      });

      $('body, html').css('height', 'auto');
    };

    Layout.prototype._max = function _max(numbers) {
      // Calculate the maximum number in a list
      var max = 0;

      Object.keys(numbers).forEach(function (key) {
        if (numbers[key] > max) {
          max = numbers[key];
        }
      });

      return max;
    };

    // Static

    Layout._jQueryInterface = function _jQueryInterface(operation) {
      return this.each(function () {
        var data = $(this).data(DATA_KEY);

        if (!data) {
          data = new Layout(this);
          $(this).data(DATA_KEY, data);
        }

        if (operation) {
          data[operation]();
        }
      });
    };

    return Layout;
  }();

  /**
   * Data API
   * ====================================================
   */


  $(window).on('load', function () {
    Layout._jQueryInterface.call($('body'));
  });

  /**
   * jQuery API
   * ====================================================
   */

  $.fn[NAME] = Layout._jQueryInterface;
  $.fn[NAME].Constructor = Layout;
  $.fn[NAME].noConflict = function () {
    $.fn[NAME] = JQUERY_NO_CONFLICT;
    return Layout._jQueryInterface;
  };

  return Layout;
}(jQuery);

/**
 * --------------------------------------------
 * AdminLTE Widget.js
 * License MIT
 * --------------------------------------------
 */

var Widget = function ($) {
  /**
   * Constants
   * ====================================================
   */

  var NAME = 'Widget';
  var DATA_KEY = 'lte.widget';
  var EVENT_KEY = '.' + DATA_KEY;
  var JQUERY_NO_CONFLICT = $.fn[NAME];

  var Event = {
    EXPANDED: 'expanded' + EVENT_KEY,
    COLLAPSED: 'collapsed' + EVENT_KEY,
    REMOVED: 'removed' + EVENT_KEY
  };

  var Selector = {
    DATA_REMOVE: '[data-widget="remove"]',
    DATA_COLLAPSE: '[data-widget="collapse"]',
    CARD: '.card',
    CARD_HEADER: '.card-header',
    CARD_BODY: '.card-body',
    CARD_FOOTER: '.card-footer',
    COLLAPSED: '.collapsed-card'
  };

  var ClassName = {
    COLLAPSED: 'collapsed-card'
  };

  var Default = {
    animationSpeed: 'normal',
    collapseTrigger: Selector.DATA_COLLAPSE,
    removeTrigger: Selector.DATA_REMOVE
  };

  var Widget = function () {
    function Widget(element, settings) {
      classCallCheck(this, Widget);

      this._element = element;
      this._parent = element.parents(Selector.CARD).first();
      this._settings = $.extend({}, Default, settings);
    }

    Widget.prototype.collapse = function collapse() {
      var _this = this;

      this._parent.children(Selector.CARD_BODY + ', ' + Selector.CARD_FOOTER).slideUp(this._settings.animationSpeed, function () {
        _this._parent.addClass(ClassName.COLLAPSED);
      });

      var collapsed = $.Event(Event.COLLAPSED);

      this._element.trigger(collapsed, this._parent);
    };

    Widget.prototype.expand = function expand() {
      var _this2 = this;

      this._parent.children(Selector.CARD_BODY + ', ' + Selector.CARD_FOOTER).slideDown(this._settings.animationSpeed, function () {
        _this2._parent.removeClass(ClassName.COLLAPSED);
      });

      var expanded = $.Event(Event.EXPANDED);

      this._element.trigger(expanded, this._parent);
    };

    Widget.prototype.remove = function remove() {
      this._parent.slideUp();

      var removed = $.Event(Event.REMOVED);

      this._element.trigger(removed, this._parent);
    };

    Widget.prototype.toggle = function toggle() {
      if (this._parent.hasClass(ClassName.COLLAPSED)) {
        this.expand();
        return;
      }

      this.collapse();
    };

    // Private

    Widget.prototype._init = function _init(card) {
      var _this3 = this;

      this._parent = card;

      $(this).find(this._settings.collapseTrigger).click(function () {
        _this3.toggle();
      });

      $(this).find(this._settings.removeTrigger).click(function () {
        _this3.remove();
      });
    };

    // Static

    Widget._jQueryInterface = function _jQueryInterface(config) {
      return this.each(function () {
        var data = $(this).data(DATA_KEY);

        if (!data) {
          data = new Widget($(this), data);
          $(this).data(DATA_KEY, typeof config === 'string' ? data : config);
        }

        if (typeof config === 'string' && config.match(/remove|toggle/)) {
          data[config]();
        } else if ((typeof config === 'undefined' ? 'undefined' : _typeof(config)) === 'object') {
          data._init($(this));
        }
      });
    };

    return Widget;
  }();

  /**
   * Data API
   * ====================================================
   */

  $(document).on('click', Selector.DATA_COLLAPSE, function (event) {
    if (event) {
      event.preventDefault();
    }

    Widget._jQueryInterface.call($(this), 'toggle');
  });

  $(document).on('click', Selector.DATA_REMOVE, function (event) {
    if (event) {
      event.preventDefault();
    }

    Widget._jQueryInterface.call($(this), 'remove');
  });

  /**
   * jQuery API
   * ====================================================
   */

  $.fn[NAME] = Widget._jQueryInterface;
  $.fn[NAME].Constructor = Widget;
  $.fn[NAME].noConflict = function () {
    $.fn[NAME] = JQUERY_NO_CONFLICT;
    return Widget._jQueryInterface;
  };

  return Widget;
}(jQuery);

exports.ControlSidebar = ControlSidebar;
exports.Layout = Layout;
exports.Treeview = Treeview;
exports.Widget = Widget;

Object.defineProperty(exports, '__esModule', { value: true });

})));
