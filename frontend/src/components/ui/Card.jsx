function Card({ children, className = "", ...props }) {
  return (
    <div 
      className={`bg-white shadow-sm rounded-lg border border-gray-200 ${className}`} 
      {...props}
    >
      {children}
    </div>
  );
}

export default Card;
