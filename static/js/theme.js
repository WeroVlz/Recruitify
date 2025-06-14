// Theme switcher functionality
document.addEventListener('DOMContentLoaded', function() {
    const toggleSwitch = document.querySelector('#checkbox');
    const themeIcon = document.querySelector('.theme-icon i');
    
    // Apply theme on page load
    const prefersDarkScheme = window.matchMedia("(prefers-color-scheme: dark)");
    const currentTheme = localStorage.getItem("theme");
    
    if (currentTheme === "dark") {
        document.documentElement.setAttribute("data-theme", "dark");
        if (toggleSwitch) toggleSwitch.checked = true;
        if (themeIcon) {
            themeIcon.classList.remove("fa-moon");
            themeIcon.classList.add("fa-sun");
        }
    } else if (currentTheme === "light") {
        document.documentElement.setAttribute("data-theme", "light");
        if (toggleSwitch) toggleSwitch.checked = false;
    } else if (prefersDarkScheme.matches) {
        document.documentElement.setAttribute("data-theme", "dark");
        if (toggleSwitch) toggleSwitch.checked = true;
        if (themeIcon) {
            themeIcon.classList.remove("fa-moon");
            themeIcon.classList.add("fa-sun");
        }
    }
    
    // Theme switcher function
    function switchTheme(e) {
        if (e.target.checked) {
            document.documentElement.setAttribute('data-theme', 'dark');
            if (themeIcon) {
                themeIcon.classList.remove('fa-moon');
                themeIcon.classList.add('fa-sun');
            }
            localStorage.setItem('theme', 'dark');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
            if (themeIcon) {
                themeIcon.classList.remove('fa-sun');
                themeIcon.classList.add('fa-moon');
            }
            localStorage.setItem('theme', 'light');
        }
    }
    
    // Add event listener for theme switch
    if (toggleSwitch) {
        toggleSwitch.addEventListener('change', switchTheme, false);
    }
});
