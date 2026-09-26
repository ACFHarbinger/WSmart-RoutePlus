(function() {
    /* --- Persistent CSS override ------------------------------------------ */
    /* CSS fill property with !important beats SVG presentation attributes
       Plotly sets on re-render, permanently overriding the white active-button
       fill. */
    var s = document.createElement('style');
    s.textContent = [
        /* Force ALL buttons dark — overrides Plotly's white active-button fill */
        'g.updatemenu-button rect { fill: __BTN_BG__ !important; stroke: __BTN_BORDER__ !important; }',
        'g.updatemenu-button text { fill: __BTN_FG__ !important; }',
        /* Active button gets bright blue */
        'g.updatemenu-button.ps-active rect { fill: #3355cc !important; }',
        /* Hover */
        'g.updatemenu-button.ps-hover rect { fill: #5577ee !important; }',
    ].join('\n');
    document.head.appendChild(s);

    /* --- Active-button tracker -------------------------------------------- */
    var activeIdx = 0;  /* 0 = first button is the default active button */

    function applyActive() {
        var gd = document.getElementById('__DIV_ID__');
        if (!gd) return;
        var btns = Array.from(gd.querySelectorAll('g.updatemenu-button'));
        if (!btns.length) return;
        btns.forEach(function(btn, i) {
            btn.classList.toggle('ps-active', i === activeIdx);
            if (!btn._psHooked) {
                btn._psHooked = true;
                btn.addEventListener('mouseenter', function() { btn.classList.add('ps-hover'); });
                btn.addEventListener('mouseleave', function() { btn.classList.remove('ps-hover'); });
            }
        });
    }

    /* Wire Plotly button-click event to update active index */
    var _wire = function() {
        var gd = document.getElementById('__DIV_ID__');
        if (!gd || !gd.on) { setTimeout(_wire, 300); return; }
        gd.on('plotly_buttonclicked', function(data) {
            /* data.active = the new active button index within the clicked menu */
            activeIdx = (typeof data.active === 'number') ? data.active : 0;
            setTimeout(applyActive, 60);
        });
        applyActive();
    };

    /* Run on load and after short delay (Plotly renders asynchronously) */
    setTimeout(_wire, 400);
    window.addEventListener('load', function() { setTimeout(_wire, 100); });
})();
