function Button({ 
  children, 
  variant = "primary", 
  size = "md", 
  className = "", 
  disabled = false, 
  ...props 
}) {
  const baseClasses = "inline-flex items-center font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors";
  
  const variants = {
    primary: "bg-primary-600 text-white hover:bg-primary-700 focus:ring-primary-500",
    secondary: "bg-gray-200 text-gray-900 hover:bg-gray-300 focus:ring-gray-500", 
    outline: "border border-gray-300 text-gray-700 bg-white hover:bg-gray-50 focus:ring-primary-500",
    danger: "bg-red-600 text-white hover:bg-red-700 focus:ring-red-500"
  };
  
  const sizes = {
    sm: "px-3 py-2 text-sm",
    md: "px-4 py-2 text-sm", 
    lg: "px-6 py-3 text-base"
  };
  
  const disabledClasses = disabled ? "opacity-50 cursor-not-allowed" : "";
  
  return (
    <button
      className={`${baseClasses} ${variants[variant]} ${sizes[size]} ${disabledClasses} ${className}`}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  );
}

export default Button;
